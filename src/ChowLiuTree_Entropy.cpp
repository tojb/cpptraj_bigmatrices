#include <memory>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <unordered_map>
#include <cstring> //memset()
#include <cassert>

#include <queue>
#include <iostream>


#include "CpptrajStdio.h"      // mprintf(), mprinterr()
#include "DataSet.h"
#include "DataSet_1D.h"

#include "ChowLiuTree_Entropy.h"
#include "Analysis_EntropyHD.h"

#undef  NDEBUG
#define __DBG
#ifdef DBG
#define CL_HEAP_POKE(); do { \
  void* _p = malloc(16); \
  if (!_p) abort(); \
  memset(_p, 0xcc, 16); \
  free(_p); \
  fprintf(stderr, "CL @ %s:%d\n", __FILE__, __LINE__); \
  fflush(stderr); \
} while(0)
#else
#define CL_HEAP_POKE(); 
#endif


/////////Initialisation:
//this initialises based on (trivial) data block
//definitions at level 0.
TreeNode* ChowLiuTree::add_leaf_node(const std::vector<size_t>& dofs, size_t id, double S_gauss) {

  //create a node and append to a list.
  nodes_.emplace_back();
  TreeNode* n = &nodes_.back();

  n->id = id;

  //init with no connections: "orphan leaf"
  n->parent = nullptr;
  n->child1 = nullptr;
  n->child2 = nullptr;
  n->tree_level = 0;

  //save what degrees of freedom (matrix rows) it controls
  n->dofs = dofs;
   
  //a million definitions of entropy.
  n->S_ng_1d = 0.0;
  n->S_ng_2d = 0.0;
  n->deltaH  = 0.0;
  n->S_gauss = S_gauss;

  return n;
}

//merge two child nodes to make a parent node.
TreeNode* ChowLiuTree::merge(TreeNode* a, TreeNode* b) {

  // allocate parent node
  nodes_.emplace_back();
  TreeNode* p = &nodes_.back();

  // structural wiring
  p->parent = nullptr;
  p->child1 = a;
  p->child2 = b;
  p->id     = nodes_.size();

  //make sure the next level is one up from previous.
  p->tree_level = a->tree_level + 1;
  if ( p->tree_level < b->tree_level + 1) {
    p->tree_level = b->tree_level + 1;
  }

  a->parent = p;
  b->parent = p;

  // merge DOFs
  p->dofs.reserve(a->dofs.size() + b->dofs.size());
  p->dofs.insert(p->dofs.end(), a->dofs.begin(), a->dofs.end());
  p->dofs.insert(p->dofs.end(), b->dofs.begin(), b->dofs.end());

  // entropy bookkeeping
  p->S_ng_1d = 0.0;
  p->S_ng_2d = 0.0;
  p->deltaH  = 0.0;

  return p;
}


//Build a graph based on pairwise distance between atoms in space:
//used to initialise the Chow-Liu with a sparse 
//approximation to the fully connected network
//aiming to preserve close nodes that will probably have high MI 
std::vector<Edge>
build_weighted_contact_graph(
    std::vector<TreeNode *> nodes,
    const double           *coords,
    size_t                  nAtoms,
    const MatrixView&       Cfull,
    double r_cut,        // Angstrom
    double w_min,       // prune weak edges
    double eps  
) {

  const double rc2 = r_cut * r_cut;

  // norms of atomic covariance blocks
  std::vector<double> block_norm(nodes.size(), 0.0);

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < nodes.size(); ++i) {
    double             s = 0.0;
    const TreeNode *node = nodes[i];
    const auto&     dofs = node->dofs;
    const size_t      nd = dofs.size();

    //per-atom covariance scale. nd:=3 in current sane use.
    for (size_t a = 0; a < nd; ++a) {
      const size_t ia = dofs[a];
      for (size_t b = 0; b < nd; ++b) {
        double x   = Cfull(ia, dofs[b]);
               s  += x * x;
      }
    }
    block_norm[i] = std::sqrt(s);
  }

  // prepare thread‑local edge buffers
  const int nthreads = omp_get_max_threads();
  std::vector<std::vector<Edge>> edges_tls(nthreads);

  //build graph. Could make this faster with
  //a neighbour list but easier just to parallelise
  #pragma omp parallel
  {
     const int tid = omp_get_thread_num();
     auto& edges_local = edges_tls[tid];

     #pragma omp for schedule(dynamic, 16)
     for (size_t i = 0; i < nodes.size() - 1; ++i) {

       const TreeNode *node = nodes[i];
       const size_t      ai = node->dofs[0]; //

       std::vector <Edge> nebs;
       nebs.clear();
       for (size_t j = i+1; j < nodes.size(); ++j) {

         const TreeNode* nj = nodes[j];
         const size_t aj = nj->dofs[0];

         //disallow neighbours outside cutoff
         const double dx = coords[ai]   - coords[aj];
         const double dy = coords[ai+1] - coords[aj+1];
         const double dz = coords[ai+2] - coords[aj+2];
         const double r2 = dx*dx + dy*dy + dz*dz;
//         if (r2 > rc2)  graph needs to be fully connected
//           continue;

         //prepare a weight which drops rapidly from 1, to zero at cutoff
         const double geom_w = std::exp(-r2 / rc2) - std::exp(-1);

         //atom-pair block covariance
         double s = 0.0;
         for (size_t a = 0; a < 3; ++a) {
           for (size_t b = 0; b < 3; ++b) {
             double x = Cfull(ai + a, aj + b);
             s += x * x;
           }
         }
         const double cov_norm = std::sqrt(s);

         //a second weight which also maxes out at 1. (?)
         const double cov_w =
               cov_norm / std::sqrt(block_norm[i]*block_norm[j] + eps);

         //save connections which are both close and covariant.
         const double w = geom_w * cov_w;

	 //keep only K nearest neighbours.
         size_t K = 16; 
         if (nebs.size() >= K && w <= nebs.back().weight) continue;
         size_t pos = nebs.size();
	 //careful: outer loop is over i.
         for (size_t ii = 0; ii < nebs.size(); ++ii) {
           if ( w > nebs[ii].weight ) {
             pos = ii;
             break;
           }
	 }
         nebs.emplace(nebs.begin() + pos,  node, nj, w);//C++ takes constructor as variable-length argmuents
         if (nebs.size() > K) nebs.pop_back();
      }
      for ( size_t i_edge = 0; i_edge < nebs.size(); i_edge++ )
         edges_local.push_back(nebs[i_edge]);
    }
  }
  // concatenate thread‑local buffers
  std::vector<Edge> edges;
  size_t total = 0;
  for (const auto& v : edges_tls)
    total += v.size();
  edges.reserve(total);

  //collecting thread buffers.
  for (auto& v : edges_tls)
    edges.insert(edges.end(), v.begin(), v.end());

  return edges;
}


//build the next level of the Chow-Liu Tree
void ChowLiuTree::greedy_merge_from_candidates(
  std::vector<TreeNode*>&     active,
  const std::vector<Edge>&    candidates
)
{
  const size_t n = active.size();

  //track which nodes were visited
  for (TreeNode* node : active) {
    node->state_flag = 0;
  }

  //loop over candidates, already should be sorted
  //by MI descending
  std::vector<TreeNode*> next_active;
  next_active.reserve(n);
  for (const auto& c : candidates) {
    auto* a = const_cast<TreeNode*>(c.a); //these pointers are constants until they aren't
    auto* b = const_cast<TreeNode*>(c.b);


    if ( a->state_flag != 0 || b->state_flag != 0 ) {
      continue;
    }

    //create a new parent node.
    TreeNode* parent = merge(a, b);
    a->state_flag = 1; //this "1" indicates "yes I have already been merged".
    b->state_flag = 1;
    next_active.push_back(parent);
  }

  // carry forward unmerged nodes
  for (TreeNode* node : active) {
    if ( node->state_flag == 0 ) {
      next_active.push_back(node);
    }
  }
  active.swap(next_active);
}

  
//debug printout.
void ChowLiuTree::debug_print_tree() const
{
  if (nodes_.empty()) {
    std::cerr << "CLTree is empty\n";
    return;
  }

  // assume: last node is the (unique) root
  const TreeNode* root = &nodes_.back();

  std::queue<const TreeNode*> q;
  q.push(root);
  std::cerr << "=== CLTree printout ===\n";

  while (!q.empty()) {
    const TreeNode* n = q.front();
    q.pop();

    std::cerr << "Node id=" << n->id;

    if (n->parent)
      std::cerr << " parent=" << n->parent->id;
    else
      std::cerr << " parent=NONE";

    if (n->child1)
      std::cerr << " child1=" << n->child1->id;
    else
      std::cerr << " child1=NONE";

    if (n->child2)
      std::cerr << " child2=" << n->child2->id;
    else
      std::cerr << " child2=NONE";

    std::cerr << "\n";

    if (n->child1) q.push(n->child1);
    if (n->child2) q.push(n->child2);
  }

  std::cerr << "=== end CLTree printout ===\n";
}


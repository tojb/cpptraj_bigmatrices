#ifndef CHOWLIUTREE_ENTROPY_H
#define CHOWLIUTREE_ENTROPY_H

#include <vector>
#include <memory>
#include <queue>
#include <functional>

//a convenient matrix accessor (overloaded for const + mutable)
struct MatrixView {
    double* data;
    size_t n;
    inline double& operator()(size_t r, size_t c) {
        return data[r*n + c];
    }
    inline const double& operator()(size_t r, size_t c) const {
        return data[r*n + c];
    }
};

//below are more specific data structures for the Chow-Liu algorithm

//make some forward declarations so that TreeNodes can refer to TreeEdges and vice-versa.
struct TreeNode;
typedef struct TreeNode {
    int       id;
    int       tree_level;
    short int state_flag; //use this for whatever.

    // structural pointers
    struct TreeNode* parent;
    struct TreeNode* child1;
    struct TreeNode* child2;

    // payload
    std::vector<size_t> dofs;

    // entropy bookkeeping
    double S_gauss;
    double S_ng_1d;
    double S_ng_2d;
    double deltaH;
} CLNode;


//a convenience structure, for generic graph building.
struct Edge {
  const TreeNode *a;
  const TreeNode *b; // tree nodes
  double    weight; // a link weight, use as convenient.

  //overload constructor.
  Edge() : a(nullptr), b(nullptr), weight(0.0) {}
  Edge(const TreeNode* a_, const TreeNode* b_) : a(a_), b(b_) {}
  Edge(const TreeNode* a_, const TreeNode* b_, double w_) : a(a_), b(b_), weight(w_) {}

};


////////////////////main class to hold and share tree information

class ChowLiuTree {

public:
  ChowLiuTree() = default;

  // Level‑0 node construction
  TreeNode* add_leaf_node(const std::vector<size_t>& dofs, size_t id, double S_gauss);
  
  // Build tree topology from MI candidates
  void build_from_candidates(
    const std::vector<Edge>& candidates
  );   

  // Build next level using lineage + new blocks
  void project_to_next_level(
    const std::vector<std::pair<size_t,size_t>>& lineage,
    const std::vector<std::vector<size_t>>&      next_blocks
  );

  //access the nodes list
  std::deque<TreeNode>& nodes() noexcept {return nodes_; }

  //merge two nodes to create the next layer up of the tree
  TreeNode* merge(TreeNode* a, TreeNode* b);

  //process a sorted list of merge candidates and merge up to the next level of the tree.
  void greedy_merge_from_candidates( std::vector<TreeNode*>&  active,
                                     const std::vector<Edge>& candidates );

  //BFS over the tree and print status.
  void debug_print_tree() const;

private:
  std::deque<TreeNode>                   nodes_;

  //unpack and make explicit b-tree (or partial, b-forest) structure.
  void root_tree(size_t root_id = 0);
};

//this convenience structure holds parameters for a subgraph merge operation
struct MergeParams {
    double min_edge_weight = 0.05;   // contact-graph threshold
    double min_mi_gain     = 1e-6;   // minimum MI (log-det units) to accept merge
    double eps_pivot       = 1e-12;  // Cholesky pivot tolerance
};

//this is some boilerplate code to find a disjoint union
//see Cormen, Leiserson, Rivest 2009.
struct UnionFind {
  std::vector<size_t> parent;
  std::vector<uint8_t> rank;

  explicit UnionFind(size_t n) : parent(n), rank(n, 0) {
    for (size_t i = 0; i < n; ++i) parent[i] = i;
  }

  //recursive search.
  size_t find(size_t x) {
    if (parent[x] != x) parent[x] = find(parent[x]); // path compression
    return parent[x];
  }

  //search and merge if not already merged.
  bool unite(size_t a, size_t b) {
    a = find(a);
    b = find(b);
    if (a == b) return false;

    // union by rank
    if (rank[a] < rank[b])  std::swap(a, b);
    parent[b] = a;
    if (rank[a] == rank[b]) ++rank[a];

    return true;
  }
};

//some MI tree graph bookkeeping
std::vector<Edge>
build_weighted_contact_graph(
    std::vector<TreeNode *> nodes,
    const double           *coords,
    size_t                  nAtoms,
    const       MatrixView& Cfull,
    double r_cut = 8.0,        // Angstrom
    double w_min = 0.05,       // prune weak edges
    double eps   = 1e-12
);

std::vector<Edge> 
    project_edges(const std::vector<Edge>& old_edges, const std::vector<size_t>& block_map);

std::vector<size_t> 
    build_block_map(size_t old_block_count, const std::vector<std::pair<size_t, size_t>>& lineage);

std::vector<std::vector<size_t>> 
    rebuild_blocks_from_lineage(const std::vector<std::vector<size_t>>&       blocks, 
		                const std::vector<std::pair<size_t, size_t>>& lineage);


void greedy_merge_from_candidates(std::vector<TreeNode*>&   active,
                                  const std::vector<Edge>&  candidates); //candidate edges sorted by MI descending




#endif

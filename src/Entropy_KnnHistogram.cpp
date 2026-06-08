#include "Entropy_KnnHistogram.h"

#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <omp.h>

//
// K nearest neighbours entropy estimator
// 
// : this is effective and standard for reasonably low dimensions,
// *providing* that we don't get got by time correlations, which will lead
// to a systematic underestimate of the entropy (overestimate of correction relative to Gaussian).
//
// The basic hack to get rid of time correlation effects is to only consider pairs of points
// outside a given window of time between each other, the "Theiler window". This is better than
// nothing however in MD, time correlations have a long tail.
//
//https://arxiv.org/pdf/1602.07440
//
//
// NB a lot of this code is AI generated which is fair enough for a standard algo but I don't
// claim that I like it.
//
// When I've had a good look at this, and tested it against reference implementations
// then I'll reassess.
//
//
//
//
//
//
//


// compute mean. easier just to write this than to work out where to import it from.
inline double mean(const std::vector<double>& x)
{
  double s = 0.0;
  for (double v : x) s += v;
  return s / (double)x.size();
}

// compute variance. local scoped inline.
inline double variance(const std::vector<double>& x, double m)
{
  double d, v = 0.0;
  for (double xi : x) {
    d  = xi - m;
    v +=  d * d;
  }
  return v / (double)x.size();
}

// "Tau" estimator: attempt to detoxify correlations
// which are inevitably present in MD timeseries:
// do not throw away correlated samples (they are all correlated)
// but instead correct for the bias that they introduce.
double estimate_tau_int_from_timeseries( const std::vector<double>& z )
{
  size_t n = z.size();
  if (n < 2)
    return 1.0;

  double m = mean(z);
  double var = variance(z, m);

  if (var <= 0.0)
    return 1.0;

  double tau = 1.0;

  // compute autocorrelation and integrate
  for (size_t lag = 1; lag < n / 2; ++lag) {

    double c = 0.0;

    for (size_t t = 0; t < n - lag; ++t) {
      c += (z[t] - m) * (z[t + lag] - m);
    }

    c /= (double)(n - lag);
    double rho = c / var;

    // stop when correlation disappears or becomes negative
    if (rho <= 0.0)
      break;

    tau += 2.0 * rho;
  }

  return tau;
}


double estimate_tau_from_projection(
  const double* Y,
  size_t n,
  size_t d)
{
  std::vector<double> z(n, 0.0);

  for (size_t i = 0; i < n; ++i) {
    const double* yi = Y + i*d;

    double s = 0.0;
    for (size_t a = 0; a < d; ++a)
      s += yi[a] * yi[a];

    z[i] = s;
  }

  return estimate_tau_int_from_timeseries(z);
}

namespace {

/* Euler–Mascheroni constant */
constexpr double EULER_GAMMA =
  0.577215664901532860606512090082402431;

/* Volume of the unit ball in R^d */
inline double unit_ball_volume(std::size_t d) {
  return std::pow(M_PI, 0.5 * d) / std::tgamma(0.5 * d + 1.0);
}

//operator to key into a cell list
struct CellKey {
  std::vector<int> idx;
  bool operator==(const CellKey& other) const {
    return idx == other.idx;
  }
};

//hashing function as an operator
struct CellKeyHash {
  std::size_t operator()(const CellKey& key) const {
    std::size_t h = 1469598103934665603ULL; // FNV-1a base
    for (int v : key.idx) {
      h ^= std::hash<int>{}(v);
      h *= 1099511628211ULL;
    }
    return h;
  }
};

//actual cell list: no need to make our own datataype, just use 
//an unordered_map.
using CellList =
  std::unordered_map<CellKey, std::vector<std::size_t>, CellKeyHash>;

static void build_cell_list(
  const double* X,          // row-major [n × d]
  std::size_t   n,
  std::size_t   d,
  double        cell_size,
  CellList&     cells
)
{
  cells.clear();
  cells.reserve(n);

  for (std::size_t i = 0; i < n; ++i) {
    const double* xi = X + i*d;

    //calculate the hash key as a vector of integer cell indices.
    CellKey key;
    key.idx.resize(d);
    for (std::size_t a = 0; a < d; ++a)
      key.idx[a] = static_cast<int>(std::floor(xi[a] / cell_size));
    cells[key].push_back(i);
  }
}





struct KDNode {
  size_t index;       // index into X
  size_t axis;
  KDNode* left;
  KDNode* right;
};

// simple squared distance
static inline double dist2(
  const double* a,
  const double* b,
  size_t d)
{
  double s = 0.0;
  for (size_t i = 0; i < d; i++) {
    double dx = a[i] - b[i];
    s += dx * dx;
  }
  return s;
}

// build kd-tree recursively
static KDNode* build_tree(
  std::vector<size_t>& idx,
  const double* X,
  size_t d,
  size_t depth)
{
  if (idx.empty())
    return nullptr;

  size_t axis = depth % d;
  size_t mid = idx.size() / 2;

  std::nth_element(idx.begin(), idx.begin() + mid, idx.end(),
    [&](size_t a, size_t b) {
      return X[a*d + axis] < X[b*d + axis];
    });

  KDNode* node = new KDNode;
  node->index = idx[mid];
  node->axis = axis;

  std::vector<size_t> left(idx.begin(), idx.begin() + mid);
  std::vector<size_t> right(idx.begin() + mid + 1, idx.end());

  node->left = build_tree(left, X, d, depth + 1);
  node->right = build_tree(right, X, d, depth + 1);

  return node;
}

// max-heap of distances (size k)
struct KHeap {
  size_t k;
  std::vector<double> vals;

  KHeap(size_t kk) : k(kk) {}

  double worst() const {
    if (vals.size() < k)
      return std::numeric_limits<double>::infinity();
    return vals.front();
  }

  void push(double v) {
    if (vals.size() < k) {
      vals.push_back(v);
      if (vals.size() == k)
        std::make_heap(vals.begin(), vals.end());
    } else if (v < vals.front()) {
      std::pop_heap(vals.begin(), vals.end());
      vals.back() = v;
      std::push_heap(vals.begin(), vals.end());
    }
  }

  double kth() const {
    if (vals.size() < k)
      return std::numeric_limits<double>::infinity();
    return vals.front();
  }
};

// recursive search
static void search_tree(
  KDNode* node,
  const double* X,
  size_t d,
  const double* query,
  size_t query_idx,
  KHeap& heap)
{
  if (!node) return;

  const double* point = X + node->index * d;

  // skip self
  if (node->index != query_idx) {
    double r2 = dist2(query, point, d);
    heap.push(r2);
  }

  size_t axis = node->axis;
  double diff = query[axis] - point[axis];

  KDNode* near = (diff < 0) ? node->left : node->right;
  KDNode* far  = (diff < 0) ? node->right : node->left;

  search_tree(near, X, d, query, query_idx, heap);

  if (diff*diff < heap.worst()) {
    search_tree(far, X, d, query, query_idx, heap);
  }
}

// cleanup
static void free_tree(KDNode* node) {
  if (!node) return;
  free_tree(node->left);
  free_tree(node->right);
  delete node;
}

// ======================================
// main function
// ======================================

void kth_neighbour_distances_kdtree(
  const double* X,
  std::size_t   n,
  std::size_t   d,
  std::size_t   k,
  std::vector<double>& eps)
{
  eps.assign(n, 0.0);

  // build index list
  std::vector<size_t> idx(n);
  for (size_t i = 0; i < n; i++)
    idx[i] = i;

  KDNode* root = build_tree(idx, X, d, 0);

  for (size_t i = 0; i < n; i++) {

    KHeap heap(k);
    const double* xi = X + i*d;

    search_tree(root, X, d, xi, i, heap);

    eps[i] = std::sqrt(heap.kth());
  }

  free_tree(root);
}



/* Compute k-th neighbour distances for an n × d block 
 *
 * where d is the degrees of freedom and n is the number of frames.
 *
 * O(n^2) scaling is not acceptable, so a neighbourlist
 * is implemented, finding nearest frames in linear time.
 *
 * */
void kth_neighbour_distances(
  const double* X,
  std::size_t   n,
  std::size_t   d,
  std::size_t   k,
  std::vector<double>& eps
)
{
  //C++ gives us various hash table constructs which can lookup in constant time
  //relative to the size of the lookup table.
  std::unordered_map<CellKey, std::vector<std::size_t>, CellKeyHash> cells;
  double cell_size;
  eps.resize(n);

  /* Estimate a reasonable cell size from RMS variance:
   * loop over frames. */
  double var = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    var += X[i*d] * X[i*d];
  var /= n;
  cell_size = std::sqrt(var) / std::pow(double(n), 1.0 / d);

  /* init the cell list */
  build_cell_list( X, n, d, cell_size, cells );

  /* stock the time series of snaps into the cells */
//#pragma omp parallel for schedule(dynamic, 16) meh. parallelise over blocks instead.
  for (std::size_t i = 0; i < n; ++i) {

    //this iteration is for frame i.
    const double* xi = X + i*d;

    std::vector<double> candidates;
    candidates.reserve( k * 8 ); //avoid reallocating if it is going to be smallish anyway

    //centre the present cell system on the present frame.
    CellKey base;
    base.idx.resize(d);
    for( size_t a = 0; a < d; ++a )
      base.idx[a] = static_cast<int>(std::floor(xi[a] / cell_size));

    //loop over 3^3 = 27 neighbour cells, in which we expect to find nearest neighbours.
    //Certainly, all neighbours inside a sphere of radius 3 x cell_size / 2 will be there.
    //
    //Treat general case of arbitrary D != 3, but expect that it is usually 3.
    //
    int shell = 0;
    while (true) {
  
      /* iterate over offsets in {-shell,…,+shell}^d :
       * this construction is basically an unroll over a nested set of d short loops,
       * each over only (-shell, 0, shell) */
      std::vector<int> offset(d, -shell);
      while (true) {
 
	//check if we did this already.
        if ( shell > 0 ){
	  bool is_inner = true;
          for (size_t a = 0; a < d; ++a) {
            if (abs(offset[a]) == shell) {
              is_inner = false;   // on the surface
              break;
            }
          }
	  //jump to next passage of the loop unless on surface of (hyper)cube.
          if ( is_inner ) {
            std::size_t a = 0;
            for (; a < d; ++a) {
              offset[a]++;
              if (offset[a] <= shell) break;
              offset[a] = -shell;
            }
            if (a == d)  break;   // finished this shell goto next_offset;
	    continue;
	  }
	}

        CellKey key;
        key.idx.resize(d); //hash key is one integer per dimension
        for (std::size_t a = 0; a < d; ++a)
          key.idx[a] = base.idx[a] + offset[a];

	//use the cell system for a lookup.
	auto it = cells.find(key);
        if (it != cells.end()) {
          for (std::size_t j : it->second) {
            if (j == i) continue;
            const double* xj = X + j*d;
            double r2 = 0.0;
	    //get Euclidean distance
            for (std::size_t a = 0; a < d; ++a) {
              const double dx = xi[a] - xj[a];
              r2 += dx * dx;
            }
            candidates.push_back(std::sqrt(r2));
          }
        }

        /* mixed‑radix increment: update the "offset" vector for next loop pass*/
	std::size_t a = 0;
        for (; a < d; ++a) {
          offset[a]++;
          if (offset[a] <= shell) break;
          offset[a] = -shell;
        }
        if (a == d)  break;   // finished this shell
      }
      
      if (candidates.size() >= k)
        break;

      //if there weren't enough neighbours (rare but possible)
      //then loop over 4^D,  5^D.. etc.
      ++shell;
    }

    std::nth_element(
      candidates.begin(),
      candidates.begin() + (k - 1),
      candidates.end()
    );

    eps[i] = candidates[k - 1];
  }
}


/* Empirical covariance determinant (small d, LU without pivot) */
double covariance_determinant(
  const double* X,
  std::size_t   n,
  std::size_t   d
)
{
  std::vector<double> C(d * d, 0.0);

#pragma omp parallel
  {
    std::vector<double> C_local(d * d, 0.0);

#pragma omp for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
      const double* xi = X + i * d;
      for (std::size_t a = 0; a < d; ++a)
        for (std::size_t b = 0; b < d; ++b)
          C_local[a * d + b] += xi[a] * xi[b];
    }

#pragma omp critical
    {
      for (std::size_t i = 0; i < d * d; ++i)
        C[i] += C_local[i];
    }
  }

  for (double& v : C)
    v /= double(n);

  /* LU determinant */
  double det = 1.0;
  std::vector<double> A = C;

  for (std::size_t i = 0; i < d; ++i) {
    const double piv = A[i * d + i];
    if (std::abs(piv) < 1e-14)
      return 0.0;

    det *= piv;

    for (std::size_t j = i + 1; j < d; ++j) {
      const double f = A[j * d + i] / piv;
      for (std::size_t k2 = i; k2 < d; ++k2)
        A[j * d + k2] -= f * A[i * d + k2];
    }
  }

  return det;
}

} // anonymous namespace


/* Exact digamma for integer arguments */
double digamma_integer(std::size_t n)
{
  if (n <= 1)
    return -EULER_GAMMA;

  double h = 0.0;
  for (std::size_t k = 1; k < n; ++k)
    h += 1.0 / double(k);

  return h - EULER_GAMMA;
}


/* kNN entropy contribution for a block */
double knn_entropy_block(
  const double* X,
  std::size_t   n,
  std::size_t   d,
  std::size_t   k
)
{
  if (d == 0 || n <= k)
    return 0.0;

  std::vector<double> eps;

  //get the local distribution of points
  kth_neighbour_distances(X, n, d, k, eps);

  //find the mean log distance
  const double eps_floor   = 1e-12; //protect the log for stability.
  double       avg_log_eps =   0.0;
  for (double e : eps)    
    avg_log_eps += std::log( std::max(e, eps_floor) );
  avg_log_eps /= double(n);


  // Kozachenko–Leonenko entropy estimator (data‑dependent part)
  //
  // H(X) = Psi(n) − Psi(k) + log V_d + d Expectation ( log eps )
  //
  // digamma_integer() gives exact Psi for integer arguments
  double H_knn =
      digamma_integer(n)
    - digamma_integer(k)
    + std::log(unit_ball_volume(d))
    + double(d) * avg_log_eps;

  //entropy correction for bunching in the k neighbours
  //due to time correlations.
  double tau = estimate_tau_from_projection(X, n, d);

  // entropy correction (if correlation time is greater than one frame).
  if ( tau > 1. ) H_knn -= 0.5 * std::log(tau);

  return H_knn;
}













#include "DataSet_MatrixDbl.h"
#include "CpptrajStdio.h"
#include <cstdlib>  // for malloc/free
void DataSet_MatrixDbl::WriteBuffer(CpptrajFile& outfile, SizeArray const& pIn) const {
  size_t x = (size_t)pIn[0];
  size_t y = (size_t)pIn[1];
  if ( x >= mat_.Ncols() || y >= mat_.Nrows() )
    outfile.Printf(format_.fmt(), 0.0);
  else 
    outfile.Printf(format_.fmt(), mat_.element(x,y));
}

double* DataSet_MatrixDbl::MatrixArray() const {
  size_t nelt = mat_.size();
  if (nelt == 0) return nullptr;

  size_t bytes = nelt * sizeof(double);

  /* use malloc so we can test for a nullptr rather than throwing */
  double* matOut = static_cast<double*>(std::malloc(bytes));
  if (!matOut) {
    mprinterr("Out of memory allocating %zu bytes for matrix\n", bytes);
    return nullptr;
  }

  double const* src = mat_.Ptr();
  if (!src) {
    mprinterr("Internal error: matrix has %zu elements but data pointer is null\n", nelt);
    free(matOut);
    return nullptr;
  }
  std::copy(src, src + nelt, matOut);
  return matOut;
}

/** Clear the matrix. */
void DataSet_MatrixDbl::Clear() {
  mat_.clear();
  vect_.clear();
  mass_.clear();
  //kind_;
  snap_ = 0;
}

/** Normalize the matrix. */
void DataSet_MatrixDbl::Normalize(double norm) {
  for (Matrix<double>::iterator it = mat_.begin(); it != mat_.end(); ++it)
    *it *= norm;
}

#ifdef MPI
int DataSet_MatrixDbl::Sync(size_t total, std::vector<int> const& rank_frames,
                            Parallel::Comm const& commIn)
{
  int total_frames = 0;
  int nframes = (int)snap_;
  commIn.ReduceMaster( &total_frames, &nframes, 1, MPI_INT, MPI_SUM );
  if (commIn.Master()) {
    snap_ = (unsigned int)total_frames;
    Darray buf( mat_.size() );
    commIn.ReduceMaster( &(buf[0]), &(mat_[0]),  mat_.size(),  MPI_DOUBLE, MPI_SUM );
    std::copy( buf.begin(), buf.end(), mat_.begin() );
    buf.assign( vect_.size(), 0.0 );
    commIn.ReduceMaster( &(buf[0]), &(vect_[0]), vect_.size(), MPI_DOUBLE, MPI_SUM );
    std::copy( buf.begin(), buf.end(), vect_.begin() );
  } else {
    commIn.ReduceMaster( 0,         &(mat_[0]),  mat_.size(),  MPI_DOUBLE, MPI_SUM );
    commIn.ReduceMaster( 0,         &(vect_[0]), vect_.size(), MPI_DOUBLE, MPI_SUM );
  }
  return 0;
}
#endif

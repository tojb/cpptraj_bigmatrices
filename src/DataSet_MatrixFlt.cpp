#include "DataSet_MatrixFlt.h"
#include "CpptrajStdio.h"

#include <cstdlib>  // for malloc/free
void DataSet_MatrixFlt::WriteBuffer(CpptrajFile& outfile, SizeArray const& pIn) const {
  size_t x = (size_t)pIn[0];
  size_t y = (size_t)pIn[1];
  if ( x >= mat_.Ncols() || y >= mat_.Nrows() )
    outfile.Printf(format_.fmt(), 0.0);
  else 
    outfile.Printf(format_.fmt(), mat_.element(x,y));
}

double* DataSet_MatrixFlt::MatrixArray() const {
  size_t nelt = mat_.size();
  if (nelt == 0) return nullptr;

  size_t bytes = nelt * sizeof(double);

  double* matOut = static_cast<double*>(std::malloc(bytes));
  if (!matOut) {
    mprinterr("Out of memory allocating %zu bytes for matrix\n", bytes);
    return nullptr;
  }

  float const* src = mat_.Ptr();
  if (!src) {
    mprinterr("Internal error: matrix has %zu elements but data pointer is null\n", nelt);
    free(matOut);
    return nullptr;
  }
  for (size_t i = 0; i < nelt; ++i)
    matOut[i] = (double)src[i];
  
  return matOut;
}

/** Clear the matrix */
void DataSet_MatrixFlt::Clear() {
  mat_.clear();
  // kind_;
}

/** Normalize the matrix */
void DataSet_MatrixFlt::Normalize(double norm) {
  for (Matrix<float>::iterator it = mat_.begin(); it != mat_.end(); ++it)
  {
    double newElt = ((double)*it) * norm;
    *it = (float)newElt;
  }
}

#ifdef MPI
int DataSet_MatrixFlt::Sync(size_t total, std::vector<int> const& rank_frames,
                            Parallel::Comm const& commIn)
{
  if (commIn.Master()) {
    std::vector<float> buf( mat_.size() );
    commIn.ReduceMaster( &(buf[0]), &(mat_[0]),  mat_.size(),  MPI_FLOAT, MPI_SUM );
    std::copy( buf.begin(), buf.end(), mat_.begin() );
  } else
    commIn.ReduceMaster( 0,         &(mat_[0]),  mat_.size(),  MPI_FLOAT, MPI_SUM );
  return 0;
}

int DataSet_MatrixFlt::SendSet(int dest, Parallel::Comm const& commIn) {
  // First send the size + type
  int size[3];
  size[0] = mat_.Ncols();
  size[1] = mat_.Nrows();
  size[2] = (int)mat_.Type();
  commIn.Send( &size, 3, MPI_INT, dest, 1700 );
  // Then send the matrix
  commIn.Send( &(mat_[0]), mat_.size(), MPI_FLOAT, dest, 1701 );
  return 0;
}

int DataSet_MatrixFlt::RecvSet(int src, Parallel::Comm const& commIn) {
  // First receive the size + type
  int size[3];
  commIn.Recv( &size, 3, MPI_INT, src, 1700 );
  if (size[2] == 0) // FULL
    mat_.resize(size[0], size[1]);
  else if (size[2] == 1) // HALF
    mat_.resize(size[0], 0);
  else // TRI
    mat_.resize(0, size[1]);
  // Then receive the matrix
  commIn.Recv( &(mat_[0]), mat_.size(), MPI_FLOAT, src, 1701);
  return 0;
}
  
#endif

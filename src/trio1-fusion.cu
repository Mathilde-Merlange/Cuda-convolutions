#include <opencv2/opencv.hpp>
#include <vector>

__global__ void trio1(unsigned char * rgb, unsigned char * g,unsigned char * t,unsigned char * x, std::size_t cols, std::size_t rows) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  if( i < cols && j < rows ) {
    g[ j * cols + i ] = (
			 307 * rgb[ 3 * ( j * cols + i ) ]
			 + 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
			 + 113 * rgb[  3 * ( j * cols + i ) + 2 ]
			 ) / 1024;
  }
  
__syncthreads();


  if ( i > 1 && j > 1 && i < cols && j < rows){
	  t[ j * cols + i ] = (
			 g[ ( (j-1) * cols + (i-1) ) ] 
			 + g[ ( (j-1) * cols + (i) ) ] 
			 + g[ ( (j-1) * cols + (i+1) ) ] 
			 + g[ ( (j) * cols + (i-1) ) ] 
			 + g[ ( (j) * cols + (i) ) ] 
			 + g[ ( (j) * cols + (i+1) ) ] 
			 + g[ ( (j+1) * cols + (i-1) ) ] 
			 + g[ ( (j+1) * cols + (i) ) ] 
			 + g[ ( (j+1) * cols + (i+1) ) ] 
			 )/9;
  }

__syncthreads();

  if( i > 1 && i < (cols - 1) && j > 1 && j < (rows - 1) )
  {
    auto hh = t[ (j-1)*cols + i - 1 ] - t[ (j-1)*cols + i + 1 ]
           + 2 * t[ j*cols + i - 1 ] - 2* t[ j*cols+i+1 ]
           + t[ (j+1)*cols + i -1] - t[ (j+1)*cols +i + 1 ];
    auto vv = t[ (j-1)*cols + i - 1 ] - t[ (j+1)*cols + i - 1 ]
           + 2 * t[ (j-1)*cols + i  ] - 2* t[ (j+1)*cols+i ]
           + t[ (j-1)*cols + i +1] - t[ (j+1)*cols +i + 1 ];

    auto res = hh * hh + vv * vv;
    res = res > 255*255 ? res = 255*255 : res;
    x[ j * cols + i ] = sqrt( (float)res );

  }
}

int main()
{
  cv::Mat m_in = cv::imread("../data/test.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;
  std::vector< unsigned char > x( rows * cols );
  cv::Mat m_out( rows, cols, CV_8UC1, x.data() );
  
  unsigned char * rgb_d;
  unsigned char * g_d;
  unsigned char * t_d;
  unsigned char * x_d;
  
  cudaMalloc( &rgb_d, 3 * rows * cols );
  cudaMalloc( &g_d, rows * cols );
  cudaMalloc( &t_d, rows * cols );
  cudaMalloc( &x_d, rows * cols );
  
  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );
  dim3 t( 32, 32 );
  dim3 b( ( cols - 1) / t.x + 1 , ( rows - 1 ) / t.y + 1 );
  
cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  cudaEventRecord( start );

trio1<<< b, t >>>( rgb_d, g_d,t_d,x_d, cols, rows );
  

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if( err != cudaSuccess )
  {
    std::cout << cudaGetErrorString( err );
  }

cudaMemcpy( x.data(), x_d, rows * cols, cudaMemcpyDeviceToHost );

cudaEventRecord( stop );
  cudaEventSynchronize( stop );

  float duration = 0.0f;
  cudaEventElapsedTime( &duration, start, stop );

  std::cout << "Total: " << duration << "ms\n";

  cv::imwrite( "trio1-cu-fusion.jpg", m_out );
  
  cudaFree( rgb_d);
  cudaFree( g_d);
  cudaFree( t_d);
  cudaFree( x_d);
  
  return 0;
}

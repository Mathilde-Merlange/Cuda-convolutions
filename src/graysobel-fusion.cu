#include <opencv2/opencv.hpp>
#include <vector>

__global__ void grayscale_sobel( unsigned char * in, unsigned char * out, std::size_t w, std::size_t h ) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  if( i < w && j < h ) {
    in[ j * w + i ] = (
			 307 * in[ 3 * ( j * w + i ) ]
			 + 604 * in[ 3 * ( j * w + i ) + 1 ]
			 + 113 * in[  3 * ( j * w + i ) + 2 ]
			 ) / 1024;
  }

  __syncthreads();

  if( i > 1 && i < (w - 1) && j > 1 && j < (h - 1) )
  {
    auto hh = in[ (j-1)*w + i - 1 ] - in[ (j-1)*w + i + 1 ]
           + 2 * in[ j*w + i - 1 ] - 2* in[ j*w+i+1 ]
           + in[ (j+1)*w + i -1] - in[ (j+1)*w +i + 1 ];
    auto vv = in[ (j-1)*w + i - 1 ] - in[ (j+1)*w + i - 1 ]
           + 2 * in[ (j-1)*w + i  ] - 2* in[ (j+1)*w+i ]
           + in[ (j-1)*w + i +1] - in[ (j+1)*w +i + 1 ];

    auto res = hh * hh + vv * vv;
    res = res > 255*255 ? res = 255*255 : res;
    out[ j * w + i ] = sqrt( (float)res );

  }
}

int main()
{
  cv::Mat m_in = cv::imread("../data/in.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  auto rows = m_in.rows;
  auto cols = m_in.cols;
  std::vector< unsigned char > g( rows * cols );
  cv::Mat m_out( rows, cols, CV_8UC1, g.data() );
  unsigned char * rgb_d;
  unsigned char * g_d;
  unsigned char * out_d;
  cudaMalloc( &rgb_d, 3 * rows * cols );
  cudaMalloc( &g_d, rows * cols );
  cudaMalloc( &out_d, rows * cols );
  cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );
  dim3 t( 32, 32 );
  dim3 b( ( cols - 1) / t.x + 1 , ( rows - 1 ) / t.y + 1 );
  grayscale_sobel<<< b, t >>>( rgb_d, g_d, cols, rows );

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if( err != cudaSuccess )
  {
    std::cout << cudaGetErrorString( err );
  }

cudaMemcpy( g.data(), g_d, rows * cols, cudaMemcpyDeviceToHost );
  cv::imwrite( "fusion.jpg", m_out );
  cudaFree( rgb_d);
  cudaFree( g_d);
  return 0;
}

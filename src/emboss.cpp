#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>

int main()
{
  cv::Mat m_in = cv::imread("../data/part1.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  
  std::vector< unsigned char > g( m_in.rows * m_in.cols );
  cv::Mat m_out( m_in.rows, m_in.cols, CV_8UC1, g.data() );

  auto start = std::chrono::system_clock::now();

  #pragma omp parallel for
  for( std::size_t j = 1 ; j < m_in.rows-1 ; ++j )
    {
      for( std::size_t i = 1 ; i < m_in.cols-1 ; ++i )
	{
	  g[ j * m_in.cols + i ] = (
			 -2 * rgb[ 3 *  ( (j-1) * m_in.cols + (i-1) ) ] 
			 - rgb[  3 * ( (j-1) * m_in.cols + (i) ) ] 
			 - rgb[  3 * ( (j) * m_in.cols + (i-1) ) ] 
			 + rgb[ 3 *  ( (j) * m_in.cols + (i+1) ) ] 
			 + rgb[  3 * ( (j+1) * m_in.cols + (i) ) ] 
			 + 2 * rgb[ 3 *  ( (j+1) * m_in.cols + (i+1) ) ] 
			 )/9;
	}
    }

  auto stop = std::chrono::system_clock::now();

  auto duration = stop - start;
  auto ms = std::chrono::duration_cast< std::chrono::milliseconds >( duration ).count();

  std::cout << ms << " ms" << std::endl;
  
  cv::imwrite( "athenes.jpg", m_out );
  
  return 0;
}

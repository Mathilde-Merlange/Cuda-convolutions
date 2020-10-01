#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>

int main()
{cv::Mat m_in = cv::imread("../data/in.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  
  std::vector< unsigned char > g( m_in.rows * m_in.cols );
  std::vector< unsigned char > h( m_in.rows * m_in.cols );
  std::vector< unsigned char > t( m_in.rows * m_in.cols );
  cv::Mat m_out( m_in.rows, m_in.cols, CV_8UC1, t.data() );

  auto start = std::chrono::system_clock::now();

  #pragma omp parallel for
  for( std::size_t j = 0 ; j < m_in.rows ; ++j )
    {
      for( std::size_t i = 0 ; i < m_in.cols ; ++i )
	{
	  g[ j * m_in.cols + i ] = (
			 307 * rgb[ 3 * ( j * m_in.cols + i ) ]
		       + 604 * rgb[ 3 * ( j * m_in.cols + i ) + 1 ]
		       + 113 * rgb[  3 * ( j * m_in.cols + i ) + 2 ]
		       ) / 1024;
	}
    }
  #pragma omp parallel for
  for( std::size_t j = 1 ; j < m_in.rows-1 ; ++j )
    {
      for( std::size_t i = 1 ; i < m_in.cols-1 ; ++i )
	{
	  h[ j * m_in.cols + i ] = (
			 g[ ( (j-1) * m_in.cols + (i-1) ) ] 
			 + g[ ( (j-1) * m_in.cols + (i) ) ] 
			 + g[ ( (j-1) * m_in.cols + (i+1) ) ] 
			 + g[ ( (j) * m_in.cols + (i-1) ) ] 
			 + g[ ( (j) * m_in.cols + (i) ) ] 
			 + g[ ( (j) * m_in.cols + (i+1) ) ] 
			 + g[ ( (j+1) * m_in.cols + (i-1) ) ] 
			 + g[ ( (j+1) * m_in.cols + (i) ) ] 
			 + g[ ( (j+1) * m_in.cols + (i+1) ) ] 
			 )/9;
	}
    }
    
    #pragma omp parallel for
    for( std::size_t j = 1 ; j < m_in.rows-1 ; ++j )
    {
      for( std::size_t i = 1 ; i < m_in.cols-1 ; ++i )
	{
	 // Horizontal
	 
	 
	int hh =     h[((j - 1) * m_in.cols + i - 1) ] -     h[((j - 1) * m_in.cols + i + 1) ]
	  + 2 * h[( j      * m_in.cols + i - 1) ] - 2 * h[( j      * m_in.cols + i + 1) ]
	  +     h[((j + 1) * m_in.cols + i - 1) ] -     h[((j + 1) * m_in.cols + i + 1) ];

	// Vertical

	int vv =     h[((j - 1) * m_in.cols + i - 1) ] -     h[((j + 1) * m_in.cols + i - 1) ]
	  + 2 * h[((j - 1) * m_in.cols + i    ) ] - 2 * h[((j + 1) * m_in.cols + i    ) ]
	  +     h[((j - 1) * m_in.cols + i + 1) ] -     h[((j + 1) * m_in.cols + i + 1) ];
	
	//std::cout << h << " h" << std::endl;
	//std::cout << v << " v" << std::endl;
	int res = hh*hh + vv*vv;
	res = res > 255*255 ? res = 255*255 : res;
	t[ j * m_in.cols + i ] = sqrt(res);
	}
    }
    

  auto stop = std::chrono::system_clock::now();

  auto duration = stop - start;
  auto ms = std::chrono::duration_cast< std::chrono::milliseconds >( duration ).count();

  std::cout << ms << " ms" << std::endl;
  
  cv::imwrite( "trio1.jpg", m_out );
  
  return 0;
}

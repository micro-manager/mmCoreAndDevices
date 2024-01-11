#pragma once

#include <cstdlib>
#include <string>

class QSIImage
{
public:

  QSIImage( unsigned int maxX, unsigned int maxY ) :
      m_buffer( 0 ),
      m_maxX( 0 ),
      m_maxY( 0 ),
      m_maxSize( 0 )
  {
    m_maxX = maxX;
    m_maxY = maxY;
    m_maxSize = maxX * maxY;

    m_buffer = (unsigned short *) malloc( m_maxSize * 2 );
    memset( m_buffer, 0, m_maxSize * 2 );
  }

  ~QSIImage( void )
  {
    free( m_buffer );
  }

  unsigned short * GetBuffer()
  {
    return m_buffer;
  }



private:

  unsigned short * m_buffer;
  unsigned int m_maxX;
  unsigned int m_maxY;
  unsigned int m_maxSize;

};


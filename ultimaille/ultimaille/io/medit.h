#ifndef __MEDIT_H__
#define __MEDIT_H__

#include <vector>
#include <cstring>
#include "ultimaille/attributes.h"
#include "ultimaille/volume.h"
#include "ultimaille/surface.h"
#include "ultimaille/polyline.h"


// TODO : MEDIT DON'T CONTAIN ATTRIBUTES, ONLY A COLOR, WHICH IS YET TO BE FEATURED HERE

namespace UM {
    void write_medit(const std::string filename, const PolyLine &pl);
    void write_medit(const std::string filename, const Surface &m);
    // for boolean, see at the end of the file
    void write_medit(const std::string filename, const Volume  &m, bool hexes_GMSH_numerotation = true);

    PolyLineAttributes read_medit(const std::string filename, PolyLine   &m);
    SurfaceAttributes  read_medit(const std::string filename, Triangles  &m);
    SurfaceAttributes  read_medit(const std::string filename, Quads      &m);
    SurfaceAttributes  read_medit(const std::string filename, Polygons   &m);
    VolumeAttributes   read_medit(const std::string filename, Tetrahedra &m);
    VolumeAttributes   read_medit(const std::string filename, Hexahedra  &m); // N.B. the file is read as is. You may want to check the det of your elements afterward. 
}

// regarding hexes, geogram convention is different of the one of medit. The writer take that into account.
   // Moreover, GMSH uses a different numerotation regarding the det of the hex -> a bool manages this. 
   // geogram is :
/*
       6--------7
      /|       /|
     / |      / |
    4--------5  |
    |  |     |  |
    |  2-----|--3
    | /      | /
    |/       |/
    0--------1

*/

// medit is : <- managed automaticly
/*
       6--------7
      /|       /|
     / |      / |
    5--------4  |
    |  |     |  |
    |  2-----|--3
    | /      | /
    |/       |/
    1--------0

*/
// GMSH is : <- the bool change the determinent. <- default
/*
       4--------7
      /|       /|
     / |      / |
    5--------6  |
    |  |     |  |
    |  0-----|--3
    | /      | /
    |/       |/
    1--------2

*/


#endif // __MEDIT_H__


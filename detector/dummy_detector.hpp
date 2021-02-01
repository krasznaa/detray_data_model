// Copyright (C) 2021 Attila Krasznahorkay.
#ifndef DETRAY_DATA_MODEL_DUMMY_DETECTOR_HPP
#define DETRAY_DATA_MODEL_DUMMY_DETECTOR_HPP

// System include(s).
#include <algorithm>
#include <vector>

namespace detray {

   /// Dummy surface type
   struct dummy_surface {

      /// Constructor
      dummy_surface( int a = 0, int b = 0 ) : m_a( a ), m_b( b ) {}

      int m_a; //< Dummy variable 1
      int m_b; //< Dummy variable 2

   }; // struct dummy_surface

   /// Default vector type, using @c std::vector
   template< typename T >
   class default_vector : public std::vector< T > {
   public:
      using std::vector< T >::vector;
   };

   /// Dummy volume type
   template< template< typename > class volume_vector = default_vector >
   struct dummy_volume {

      /// Default constructor
      dummy_volume() = default;
      /// Copy constructor
      template< template< typename > class parent_volume_vector >
      dummy_volume( const dummy_volume< parent_volume_vector >& parent )
      : m_surface_indices( parent.m_surface_indices.size() ) {

         std::copy( std::begin( parent.m_surface_indices ),
                    std::end( parent.m_surface_indices ),
                    std::begin( m_surface_indices ) );
      }

      /// Vector of indices to the surfaces belonging to this volume
      volume_vector< std::size_t > m_surface_indices;

   }; // class dummy_volume

   /// PoD with pointers to all memory blocks used by the detector description
   template< template< typename > class volume_vector = default_vector >
   struct dummy_detector_data {

      /// Pointer to the start of the memory block holding all the surfaces
      const dummy_surface* m_surfaces;
      /// The number of surfaces
      std::size_t m_nSurfaces;
      /// Pointer to the start of the memory block holding all the volumes
      const dummy_volume< volume_vector >* m_volumes;
      /// The number of volumes
      std::size_t m_nVolumes;

   }; // struct dummy_detector_data

   /// "Detector" type used to experiment with vector handling
   ///
   /// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
   ///
   template< template< typename > class detector_vector = default_vector,
             template< typename > class volume_vector = default_vector >
   class dummy_detector {

   public:
      /// Make sure that all template specialisations are friends of each other
      template< template< typename > class other_detector_vector,
                template< typename > class other_volume_vector >
      friend class dummy_detector;

      /// Convenience type for the volume
      using volume = dummy_volume< volume_vector >;

      /// Default constructor
      dummy_detector() = default;

      /// Copy constructor
      template< template< typename > class parent_detector_vector,
                template< typename > class parent_volume_vector >
      dummy_detector( const dummy_detector< parent_detector_vector,
                                            parent_volume_vector >& parent )
      : m_surfaces( parent.m_surfaces.size() ),
        m_volumes( parent.m_volumes.size() ) {

         std::copy( std::begin( parent.m_surfaces ),
                    std::end( parent.m_surfaces ),
                    std::begin( m_surfaces ) );
         std::copy( std::begin( parent.m_volumes ),
                    std::end( parent.m_volumes ),
                    std::begin( m_volumes ) );
      }

      /// Constructor from a dummy data object. Only available if we use our
      /// custom vector type.
      dummy_detector( const dummy_detector_data< volume_vector >& data )
      : m_surfaces( data.m_nSurfaces, data.m_surfaces ),
        m_volumes( data.m_nVolumes, data.m_volumes ) {}

      /// Accessor to the surface vector
      detector_vector< dummy_surface >& surfaces() { return m_surfaces; }
      /// Accessor to the volume vector
      detector_vector< volume >& volumes() { return m_volumes; }

      /// Function generating a PoD with the detector data
      dummy_detector_data< volume_vector > data() const {
         return { m_surfaces.data(), m_surfaces.size(),
                  m_volumes.data(), m_volumes.size() };
      }

   private:
      detector_vector< dummy_surface > m_surfaces; ///< The detector surfaces
      detector_vector< volume > m_volumes; ///< The detector volumes

   }; // class dummy_detector

} // namespace detray

#endif // DETRAY_DATA_MODEL_DUMMY_DETECTOR_HPP

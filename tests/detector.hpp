/** Detray Data Model project, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */
#pragma once

// Local include(s).
#include "detraydm/utils/cuda_types.hpp"
#include "detraydm/containers/host_vector.hpp"

// System include(s).
#include <algorithm>
#include <vector>

namespace detraydm::test {

   /// Dummy surface type
   struct surface {
      int m_a = 0; //< Dummy variable 1
      int m_b = 0; //< Dummy variable 2
   }; // struct surface

   /// Dummy volume type
   template< template< typename > class vector_type = host_vector >
   struct volume {

      /// Default constructor
      volume() = default;
      /// Copy constructor
      template< template< typename > class parent_vector_type >
      volume( const volume< parent_vector_type >& parent )
      : m_surface_indices( parent.m_surface_indices.size() ) {

         std::copy( std::begin( parent.m_surface_indices ),
                    std::end( parent.m_surface_indices ),
                    std::begin( m_surface_indices ) );
      }

      /// Vector of indices to the surfaces belonging to this volume
      vector_type< std::size_t > m_surface_indices;

   }; // class volume

   /// PoD with pointers to all memory blocks used by the detector description
   template< template< typename > class volume_vector_type = host_vector >
   struct detector_data {

      /// Type for the surface objects in memory
      typedef surface surface_type;
      /// Type of the volume objects in memory
      typedef volume< volume_vector_type > volume_type;

      /// Pointer to the start of the memory block holding all the surfaces
      const surface_type* m_surfaces;
      /// The number of surfaces
      std::size_t m_nSurfaces;
      /// Pointer to the start of the memory block holding all the volumes
      const volume_type* m_volumes;
      /// The number of volumes
      std::size_t m_nVolumes;

   }; // struct detector_data

   /// "Detector" type used to experiment with vector handling
   ///
   /// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
   ///
   template< template< typename > class detector_vector_type = host_vector,
             template< typename > class volume_vector_type = host_vector >
   class detector {

   public:
      /// Make sure that all template specialisations are friends of each other
      template< template< typename > class other_detector_vector_type,
                template< typename > class other_volume_vector_type >
      friend class detector;

      /// Convenience type for the surface
      using surface_type = surface;
      /// Convenience type for the volume
      using volume_type = volume< volume_vector_type >;

      /// Default constructor
      DETRAYDM_CUDA_HOST
      detector() {}

      /// Copy constructor
      template< template< typename > class parent_detector_vector_type,
                template< typename > class parent_volume_vector_type >
      DETRAYDM_CUDA_HOST
      detector( const detector< parent_detector_vector_type,
                                parent_volume_vector_type >& parent )
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
      DETRAYDM_CUDA_HOST_AND_DEVICE
      detector( const detector_data< volume_vector_type >& data )
      : m_surfaces( data.m_nSurfaces, data.m_surfaces ),
        m_volumes( data.m_nVolumes, data.m_volumes ) {}

      /// Assignment operator
      template< template< typename > class parent_detector_vector_type,
                template< typename > class parent_volume_vector_type >
      DETRAYDM_CUDA_HOST
      detector< detector_vector_type, volume_vector_type >&
      operator=( const detector< parent_detector_vector_type,
                                 parent_volume_vector_type >& rhs ) {

         // Prevent self-assignment.
         if( &rhs == this ) {
            return *this;
         }

         // Copy the payload.
         m_surfaces.resize( rhs.m_surfaces.size() );
         std::copy( std::begin( rhs.m_surfaces ), std::end( rhs.m_surfaces ),
                    std::begin( m_surfaces ) );
         m_volumes.resize( rhs.m_volumes.size() );
         std::copy( std::begin( rhs.m_volumes ), std::end( rhs.m_volumes ),
                    std::begin( m_volumes ) );

         // Return this object.
         return *this;
      }

      /// Accessor to the surface vector
      DETRAYDM_CUDA_HOST_AND_DEVICE
      detector_vector_type< surface_type >& surfaces() { return m_surfaces; }
      /// Accessor to the volume vector
      DETRAYDM_CUDA_HOST_AND_DEVICE
      detector_vector_type< volume_type >& volumes() { return m_volumes; }

      /// Function generating a PoD with the detector data
      DETRAYDM_CUDA_HOST
      detector_data< volume_vector_type > data() const {
         return { m_surfaces.data(), m_surfaces.size(),
                  m_volumes.data(), m_volumes.size() };
      }

   private:
      detector_vector_type< surface_type > m_surfaces; ///< The detector surfaces
      detector_vector_type< volume_type > m_volumes; ///< The detector volumes

   }; // class detector

} // namespace detraydm::test

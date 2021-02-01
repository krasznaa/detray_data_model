// Copyright (C) 2021 Attila Krasznahorkay.
#ifndef DETRAY_DATA_MODEL_HOST_VECTOR_HPP
#define DETRAY_DATA_MODEL_HOST_VECTOR_HPP

// System include(s).
#include <vector>

namespace detray {

   /// Vector type to use "on the host"
   ///
   /// Hiding the allocator template argument from the user. Which is necessary
   /// in the template types of this project that use a selectable vector type.
   ///
   template< typename T >
   using host_vector = std::vector< T >;

} // namespace detray

#endif // DETRAY_DATA_MODEL_HOST_VECTOR_HPP

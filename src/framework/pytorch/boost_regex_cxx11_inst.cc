// Explicitly instantiate the std::string-returning member functions of
// boost::cpp_regex_traits_implementation<char> with the CXX11 ABI
// (_GLIBCXX_USE_CXX11_ABI=1).  folly/recstore are built with the new C++ ABI,
// but the system boost 1.69 shared/static libraries were compiled with the old
// ABI and only provide the non-B5cxx11 instantiations of transform() and
// transform_primary(), which return std::basic_string<char> and are therefore
// ABI-tagged.  Instantiating them here defines the B5cxx11-mangled symbols so
// the recstore ops shared library resolves at load time.
//
// Include the umbrella header (and <locale> for std::messages) so all boost
// regex config/workaround macros are defined before the template bodies are
// parsed with GCC 11.
#include <locale>
#include <boost/regex.hpp>

template
boost::re_detail_106900::cpp_regex_traits_implementation<char>::string_type
boost::re_detail_106900::cpp_regex_traits_implementation<char>::transform(
    const char*, const char*) const;

template
boost::re_detail_106900::cpp_regex_traits_implementation<char>::string_type
boost::re_detail_106900::cpp_regex_traits_implementation<char>::transform_primary(
    const char*, const char*) const;

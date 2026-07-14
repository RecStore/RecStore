#pragma once

extern "C" void RecStoreForceLinkIoUringBackend();
#ifdef RECSTORE_ENABLE_SPDK
extern "C" void RecStoreForceLinkSpdkBackend();
#endif

inline void ForceLinkIOBackends() {
  RecStoreForceLinkIoUringBackend();
#ifdef RECSTORE_ENABLE_SPDK
  RecStoreForceLinkSpdkBackend();
#endif
}

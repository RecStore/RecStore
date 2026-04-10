#pragma once

extern "C" void RecStoreForceLinkIoUringBackend();
extern "C" void RecStoreForceLinkSpdkBackend();

inline void ForceLinkIOBackends() {
  RecStoreForceLinkIoUringBackend();
  RecStoreForceLinkSpdkBackend();
}

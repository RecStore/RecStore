#include "storage/io_backend/force_link.h"
#include "storage/kv_engine/engine_composite.h"
#include "storage/kv_engine/engine_petkv.h"

#include "gflags/gflags.h"

extern "C" void RecStoreForceLinkFasterKVEngine();
extern "C" void RecStoreForceLinkDETableEngine();
extern "C" void RecStoreForceLinkHPSEngine();

namespace {
struct IOBackendLinkGuard {
  IOBackendLinkGuard() {
    ForceLinkIOBackends();
#ifdef RECSTORE_ENABLE_DETABLE_ENGINE
    RecStoreForceLinkDETableEngine();
#endif
#ifdef RECSTORE_ENABLE_FASTERKV_ENGINE
    RecStoreForceLinkFasterKVEngine();
#endif
#ifdef RECSTORE_ENABLE_HPS_ENGINE
    RecStoreForceLinkHPSEngine();
#endif
  }
};
const IOBackendLinkGuard kIoBackendLinkGuard;
} // namespace

DEFINE_int32(prefetch_method,
             0,
             "PetKV BatchGet prefetch method: 0=single Get loop, 1=BatchGet");

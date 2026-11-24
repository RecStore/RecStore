# Offline tool: rewrite checkpoint shards written before the optimizer key
# layout recorded `optimizer_layout`. Not part of the server.
add_executable(checkpoint_migrate
    checkpoint_migrate_main.cpp
    checkpoint_migrate.cc
)
target_link_libraries(checkpoint_migrate PUBLIC
    base
    Threads::Threads
    stdc++fs
)

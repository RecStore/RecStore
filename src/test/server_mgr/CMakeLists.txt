add_library(ps_server_mgr STATIC
    ps_server_launcher.cpp
)

target_include_directories(ps_server_mgr PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(ps_server_mgr PUBLIC
    pthread
    stdc++fs
)

add_executable(ps_server_launcher_cli
    ps_server_launcher_cli.cpp
)

target_link_libraries(ps_server_launcher_cli PUBLIC
    ps_server_mgr
    pthread
    stdc++fs
)

add_executable(test_ps_server_launcher
    test_ps_server_launcher.cpp
)

target_link_libraries(test_ps_server_launcher PUBLIC
    ps_server_mgr
    gtest
    gtest_main
    pthread
    stdc++fs
)

add_test(NAME test_ps_server_launcher COMMAND test_ps_server_launcher)

add_python_unittest(
    test_ps_server_helpers_py
    src/test/scripts/test_ps_server_helpers.py
)
add_python_unittest(
    test_ps_server_runner_py
    src/test/scripts/test_ps_server_runner.py
)
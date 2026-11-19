#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <unistd.h>

#include "base/config.h"
#include "base/factory.h"

namespace {

class ScopedCurrentPath {
public:
  explicit ScopedCurrentPath(const std::filesystem::path& path)
      : old_path_(std::filesystem::current_path()) {
    std::filesystem::current_path(path);
  }

  ~ScopedCurrentPath() noexcept {
    std::error_code ec;
    std::filesystem::current_path(old_path_, ec);
  }

private:
  std::filesystem::path old_path_;
};

class ScopedTempDir {
public:
  explicit ScopedTempDir(std::filesystem::path path) : path_(std::move(path)) {
    std::filesystem::create_directories(path_);
  }

  ~ScopedTempDir() { std::filesystem::remove_all(path_); }

  const std::filesystem::path& path() const { return path_; }

private:
  std::filesystem::path path_;
};

std::filesystem::path MakeTempRoot() {
  auto root =
      std::filesystem::temp_directory_path() /
      std::filesystem::path(
          "recstore_config_test_" + std::to_string(::getpid()) + "_" +
          std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
  return root;
}

void Touch(const std::filesystem::path& path) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  out << "{}";
}

} // namespace

namespace base {

TEST(ConfigPathTest, PrefersSearchPathBeforeDefaultPath) {
  ScopedTempDir root(MakeTempRoot() / "prefers_search_path");

  const auto default_path = root.path() / "default" / "recstore_config.json";
  const auto nested_path  = root.path() / "repo" / "a" / "b";
  const auto repo_config  = root.path() / "repo" / "recstore_config.json";
  Touch(default_path);
  Touch(repo_config);
  std::filesystem::create_directories(nested_path);

  auto found = detail::FindRecStoreConfigPath(default_path, nested_path);

  ASSERT_TRUE(found.has_value());
  EXPECT_EQ(std::filesystem::weakly_canonical(*found),
            std::filesystem::weakly_canonical(repo_config));
}

TEST(ConfigPathTest, SearchesFromCurrentDirectoryUpward) {
  ScopedTempDir root(MakeTempRoot() / "searches_upward");

  const auto missing_default = root.path() / "missing" / "recstore_config.json";
  const auto nested_path     = root.path() / "repo" / "a" / "b";
  const auto repo_config     = root.path() / "repo" / "recstore_config.json";
  Touch(repo_config);
  std::filesystem::create_directories(nested_path);

  {
    ScopedCurrentPath cwd(nested_path);
    auto found = detail::FindRecStoreConfigPath(
        missing_default, std::filesystem::current_path());

    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(std::filesystem::weakly_canonical(*found),
              std::filesystem::weakly_canonical(repo_config));
  }
}

TEST(ConfigPathTest, ReturnsNulloptWhenConfigIsMissing) {
  ScopedTempDir root(MakeTempRoot() / "missing_nullopt");

  const auto missing_default = root.path() / "missing" / "recstore_config.json";
  const auto nested_path     = root.path() / "repo" / "a" / "b";
  std::filesystem::create_directories(nested_path);

  auto found = detail::FindRecStoreConfigPath(missing_default, nested_path);

  EXPECT_FALSE(found.has_value());
}

TEST(ConfigPathTest, ResolveThrowsWhenConfigIsMissing) {
  ScopedTempDir root(MakeTempRoot() / "missing_throw");

  const auto missing_default = root.path() / "missing" / "recstore_config.json";
  const auto nested_path     = root.path() / "repo" / "a" / "b";
  std::filesystem::create_directories(nested_path);

  EXPECT_THROW(detail::ResolveRecStoreConfigPath(missing_default, nested_path),
               std::runtime_error);
}

TEST(FactoryTest, UnknownKeyNamesTheKeyAndExitsImmediately) {
  EXPECT_EXIT(base::Factory<int>::NewInstance("missing-key"),
              ::testing::ExitedWithCode(255),
              "no Factory instance for key: missing-key");
}

} // namespace base

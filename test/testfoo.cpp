#include "testfoo.h"

FooTest::FooTest() {}

FooTest::~FooTest() {};

void FooTest::SetUp() {};

void FooTest::TearDown() {};

TEST_F(FooTest, ByDefaultBazTrueIsTrue) {
    EXPECT_EQ(true, true);
}

TEST_F(FooTest, ByDefaultBazFalseIsFalse) {
    EXPECT_FALSE(false);
}

TEST_F(FooTest, SometimesBazFalseIsTrue) {
    EXPECT_FALSE(true);
}

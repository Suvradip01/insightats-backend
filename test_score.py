from app.services.pipeline.orchestrator import _composite_headline_score

def test_score():
    # Test case 1: The original values from the user's screenshot
    # Skills=63, Exp=83, Proj=40, Structure=62
    s1 = _composite_headline_score(0.9, 63, 83, 40, 62)
    # Expected unblended: 0.35(63) + 0.25(83) + 0.25(40) + 0.15(62) = 62.1
    # Expected blended: 0.42 * (0.9 * 100) + 0.58 * 62.1 = 37.8 + 36.018 = 73.818 => round(73.818) = 74
    
    # Test case 2
    # Skills=64, Exp=84, Proj=40, Structure=50
    s2 = _composite_headline_score(0.9, 64, 84, 40, 50)
    # Expected unblended: 0.35(64) + 0.25(84) + 0.25(40) + 0.15(50) = 60.9
    # Expected blended: 0.42 * (0.9 * 100) + 0.58 * 60.9 = 37.8 + 35.322 = 73.122 => round(73.122) = 73

    print(f"Test 1 UI values (63/83/40/62) -> New Overall Score: {s1} (Expected 74)")
    print(f"Test 2 UI values (64/84/40/50) -> New Overall Score: {s2} (Expected 73)")

    # Assertions
    assert s1 == 74, f"Expected 74 but got {s1}"
    assert s2 == 73, f"Expected 73 but got {s2}"
    print("Scoring math is correct!")

if __name__ == "__main__":
    test_score()

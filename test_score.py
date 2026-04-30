from app.services.pipeline.orchestrator import _composite_headline_score

def test_score():
    # Test case 1: The original values from the user's screenshot
    # Skills=63, Exp=83, Proj=40, Structure=62
    s1 = _composite_headline_score(0.9, 63, 83, 40, 62)
    # Expected: 0.35(63) + 0.25(83) + 0.25(40) + 0.15(62)
    # = 22.05 + 20.75 + 10 + 9.3 = 62.1 => round(62.1) = 62
    
    # Test case 2
    # Skills=64, Exp=84, Proj=40, Structure=50
    s2 = _composite_headline_score(0.9, 64, 84, 40, 50)
    # Expected: 0.35(64) + 0.25(84) + 0.25(40) + 0.15(50)
    # = 22.4 + 21 + 10 + 7.5 = 60.9 => round(60.9) = 61

    print(f"Test 1 UI values (63/83/40/62) -> New Overall Score: {s1} (Expected 62)")
    print(f"Test 2 UI values (64/84/40/50) -> New Overall Score: {s2} (Expected 61)")

    # Assertions
    assert s1 == 62, f"Expected 62 but got {s1}"
    assert s2 == 61, f"Expected 61 but got {s2}"
    print("Scoring math is correct!")

if __name__ == "__main__":
    test_score()

"""
Comprehensive, balanced prompts for fall detection.
Equal number of fall and non-fall prompts to avoid bias.
"""

from typing import List, Dict, Tuple

# FALL PROMPTS - More specific and accurate
FALL_PROMPTS = [
    # Direct fall descriptions
    "a person falling down to the ground",
    "a person collapsing and falling",
    "a person losing balance and falling",
    "a person tripping and falling down",
    "a person falling from standing position",
    "a person falling backwards",
    "a person falling forwards",
    "a person falling sideways",
    "a person suddenly falling down",
    "a person falling and hitting the ground",
    
    # Elderly/medical falls
    "an elderly person falling down",
    "an elderly person collapsing",
    "an elderly person losing balance and falling",
    "a person fainting and falling",
    "a person having a medical emergency and falling",
    
    # Fall from height/position
    "a person falling from a bed",
    "a person falling from a chair",
    "a person falling from standing",
    "a person falling from walking",
    "a person falling from running",
    
    # Specific fall scenarios
    "a person slipping and falling",
    "a person stumbling and falling",
    "a person losing footing and falling",
    "a person falling due to imbalance",
    "a person falling after losing support",
    
    # Motion-based falls
    "a person falling while walking",
    "a person falling while running",
    "a person falling while standing",
    "a person falling while turning",
    "a person falling while getting up",
    
    # Impact/ground contact
    "a person falling and lying on the ground",
    "a person falling and remaining on the floor",
    "a person falling and not getting up",
    "a person falling and staying down",
    "a person falling and lying motionless",
    
    # Emergency falls
    "a person having a fall accident",
    "a person experiencing a fall",
    "a person falling unexpectedly",
    "a person falling without control",
    "a person falling and needing help",
    
    # Context-specific falls
    "a person falling in a room",
    "a person falling indoors",
    "a person falling in a hallway",
    "a person falling in a bedroom",
    "a person falling in a living room",
    
    # Body position during fall
    "a person falling with arms extended",
    "a person falling with body horizontal",
    "a person falling with head down",
    "a person falling with legs up",
    "a person falling in an uncontrolled manner",
    
    # Severity indicators
    "a person falling hard to the ground",
    "a person falling and not moving",
    "a person falling and lying flat",
    "a person falling and remaining still",
    "a person falling and showing distress",
    
    # Age-specific
    "an older person falling down",
    "a senior person falling",
    "an aged person collapsing",
    "an elderly individual falling",
    "a senior citizen falling down",
    
    # Activity-related falls
    "a person falling while exercising",
    "a person falling during activity",
    "a person falling while moving",
    "a person falling during daily activity",
    "a person falling while performing task",
    
    # Recovery/response
    "a person falling and unable to get up",
    "a person falling and lying helpless",
    "a person falling and requiring assistance",
    "a person falling and not recovering",
    
    # Visual characteristics
    "a person falling with body horizontal to ground",
    "a person falling with body parallel to floor",
    "a person falling in prone position",
    "a person falling in supine position",
    "a person falling with body flat",
    
    # Temporal aspects
    "a person falling suddenly",
    "a person falling quickly",
    "a person falling rapidly",
    "a person falling immediately",
    "a person falling without warning",
    
    # Environmental
    "a person falling on a hard surface",
    "a person falling on the floor",
    "a person falling on the ground",
    "a person falling on a surface",
    "a person falling onto floor",
    
    # Additional fall prompts to reach 100
    "a person falling and landing on ground",
    "a person falling and hitting floor",
    "a person falling and ending up on ground",
    "a person falling and coming to rest on floor",
    "a person falling and lying on surface",
    "a person falling and remaining on ground",
    "a person falling and staying on floor",
    "a person falling and not standing up",
    "a person falling and lying down",
    "a person falling and ending up lying",
    "a person falling and becoming horizontal",
    "a person falling and going to ground",
    "a person falling and reaching floor",
    "a person falling and contacting ground",
    "a person falling and ending on floor",
]

# NON-FALL PROMPTS - More specific and accurate
NON_FALL_PROMPTS = [
    # Normal walking
    "a person walking normally",
    "a person walking steadily",
    "a person walking upright",
    "a person walking with balance",
    "a person walking confidently",
    "a person walking at normal pace",
    "a person walking in a straight line",
    "a person walking with good posture",
    "a person walking with stability",
    "a person walking without falling",
    
    # Standing
    "a person standing upright",
    "a person standing still",
    "a person standing steadily",
    "a person standing with balance",
    "a person standing normally",
    "a person standing straight",
    "a person standing firmly",
    "a person standing stable",
    "a person standing without falling",
    "a person standing with good posture",
    
    # Sitting
    "a person sitting down",
    "a person sitting on a chair",
    "a person sitting on a bed",
    "a person sitting comfortably",
    "a person sitting normally",
    "a person sitting upright",
    "a person sitting with good posture",
    "a person sitting and not falling",
    "a person sitting safely",
    "a person sitting in a chair",
    
    # Lying down (calm)
    "a person lying down calmly",
    "a person lying on a bed",
    "a person lying comfortably",
    "a person lying still",
    "a person lying peacefully",
    "a person lying in bed",
    "a person lying down safely",
    "a person lying without distress",
    "a person lying in a relaxed position",
    "a person lying down normally",
    
    # KTH Activities - Walking
    "a person walking in a video",
    "a person walking in an outdoor setting",
    "a person walking in a controlled manner",
    "a person walking with normal gait",
    "a person walking with steady steps",
    
    # KTH Activities - Jogging
    "a person jogging",
    "a person jogging normally",
    "a person jogging steadily",
    "a person jogging with balance",
    "a person jogging without falling",
    "a person jogging in a video",
    "a person jogging outdoors",
    "a person jogging with good form",
    "a person jogging at steady pace",
    "a person jogging with stability",
    
    # KTH Activities - Running (keep 5)
    "a person running",
    "a person running normally",
    "a person running steadily",
    "a person running with balance",
    "a person running without falling",
    
    # KTH Activities - Boxing
    "a person boxing",
    "a person boxing normally",
    "a person boxing without falling",
    "a person boxing in a video",
    "a person boxing with good form",
    "a person boxing with stability",
    "a person boxing with control",
    "a person boxing with coordination",
    "a person boxing with balance",
    
    # KTH Activities - Hand Clapping
    "a person clapping hands",
    "a person hand clapping",
    "a person clapping normally",
    "a person clapping with balance",
    "a person clapping without falling",
    "a person clapping in a video",
    "a person clapping with good form",
    "a person clapping with stability",
    "a person clapping with control",
    "a person clapping with coordination",
    
    # KTH Activities - Hand Waving
    "a person waving hands",
    "a person hand waving",
    "a person waving normally",
    "a person waving with balance",
    "a person waving without falling",
    "a person waving in a video",
    "a person waving with good form",
    "a person waving with stability",
    "a person waving with control",
    "a person waving with coordination",
    
    # Movement without falling (keep 5)
    "a person moving normally",
    "a person moving with balance",
    "a person moving without falling",
    "a person moving steadily",
    "a person moving with control",
    
    # Getting up (safely) - keep 5
    "a person getting up from bed",
    "a person getting up from chair",
    "a person getting up safely",
    "a person getting up normally",
    "a person getting up with balance",
]

# Ensure equal number of prompts
assert len(FALL_PROMPTS) == len(NON_FALL_PROMPTS), \
    f"Prompt imbalance: {len(FALL_PROMPTS)} fall vs {len(NON_FALL_PROMPTS)} non-fall"

# Combined prompts (fall first, then non-fall for easy identification)
ALL_PROMPTS = FALL_PROMPTS + NON_FALL_PROMPTS

# Prompt metadata
PROMPT_INFO = {
    'num_fall_prompts': len(FALL_PROMPTS),
    'num_non_fall_prompts': len(NON_FALL_PROMPTS),
    'total_prompts': len(ALL_PROMPTS),
    'fall_prompt_indices': list(range(len(FALL_PROMPTS))),
    'non_fall_prompt_indices': list(range(len(FALL_PROMPTS), len(ALL_PROMPTS))),
    'balanced': len(FALL_PROMPTS) == len(NON_FALL_PROMPTS)
}


def get_fall_prompts() -> List[str]:
    """Get all fall-related prompts."""
    return FALL_PROMPTS.copy()


def get_non_fall_prompts() -> List[str]:
    """Get all non-fall-related prompts."""
    return NON_FALL_PROMPTS.copy()


def get_all_prompts() -> List[str]:
    """Get all prompts (fall + non-fall)."""
    return ALL_PROMPTS.copy()


def get_prompt_info() -> Dict:
    """Get information about the prompts."""
    return PROMPT_INFO.copy()


def get_balanced_prompts(num_prompts: int = None) -> Tuple[List[str], List[str]]:
    """
    Get balanced prompts.
    
    Args:
        num_prompts: Number of prompts per class. If None, returns all.
    
    Returns:
        Tuple of (fall_prompts, non_fall_prompts)
    """
    if num_prompts is None:
        return FALL_PROMPTS.copy(), NON_FALL_PROMPTS.copy()
    
    # Ensure we don't exceed available prompts
    num_prompts = min(num_prompts, len(FALL_PROMPTS))
    
    return FALL_PROMPTS[:num_prompts], NON_FALL_PROMPTS[:num_prompts]


if __name__ == "__main__":
    print("="*80)
    print("FALL DETECTION PROMPTS")
    print("="*80)
    print(f"\nFall Prompts: {len(FALL_PROMPTS)}")
    print(f"Non-Fall Prompts: {len(NON_FALL_PROMPTS)}")
    print(f"Total Prompts: {len(ALL_PROMPTS)}")
    print(f"Balanced: {PROMPT_INFO['balanced']}")
    print("\nSample Fall Prompts:")
    for i, prompt in enumerate(FALL_PROMPTS[:5], 1):
        print(f"  {i}. {prompt}")
    print("\nSample Non-Fall Prompts:")
    for i, prompt in enumerate(NON_FALL_PROMPTS[:5], 1):
        print(f"  {i}. {prompt}")


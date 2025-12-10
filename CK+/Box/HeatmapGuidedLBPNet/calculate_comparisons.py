"""
Calculate LBP comparisons per pattern for different network configurations
"""
import math

def calculate_comparisons_per_pattern(
    channels,           # List of channel counts, e.g., [1, 20, 30, 40, 40, 50, 50, 60, 60]
    spatial_dims,       # List of (H, W) tuples, e.g., [(29,18), (29,18), (15,9), ...]
    p_decay,            # List of P values for stages 1+, e.g., [8, 6, 6, 4, 4, 4, 4]
    adaptive_p_counts,  # Dict of {p_value: pixel_count} for stage 0, e.g., {4:48, 6:273, ...}
    downsample_after=None  # List of stage indices to downsample after (optional)
):
    """
    Calculate total comparisons per pattern.
    
    Args:
        channels: List of input channel counts for each stage [C0, C1, ..., CL]
        spatial_dims: List of (height, width) for each stage
        p_decay: List of P values for stages 1 onwards
        adaptive_p_counts: Dict mapping P values to pixel counts for stage 0
        downsample_after: List of stage indices after which downsampling occurs (optional)
    
    Returns:
        dict: Dictionary containing total comparisons and per-stage breakdown
    """
    
    num_stages = len(channels)
    comparisons_per_stage = {}
    
    # Stage 0: Adaptive-P
    stage0_comp = channels[0] * sum(count * p_val for p_val, count in adaptive_p_counts.items())
    comparisons_per_stage[0] = {
        'channels': channels[0],
        'spatial': spatial_dims[0],
        'p_type': 'adaptive',
        'comparisons': stage0_comp
    }
    
    # Stages 1 to L-1: P-decay
    for stage in range(1, num_stages):
        H, W = spatial_dims[stage]
        C = channels[stage]
        P = p_decay[stage - 1]  # p_decay is indexed from 0 but for stages 1+
        
        comp = H * W * C * P
        comparisons_per_stage[stage] = {
            'channels': C,
            'spatial': (H, W),
            'p_value': P,
            'comparisons': comp
        }
    
    total_comparisons = sum(stage_info['comparisons'] for stage_info in comparisons_per_stage.values())
    
    return {
        'total_comparisons': total_comparisons,
        'per_stage': comparisons_per_stage,
        'num_stages': num_stages
    }


def print_comparison_report(result):
    """Print a formatted report of the comparison calculation."""
    
    print("=" * 80)
    print("LBP COMPARISONS PER PATTERN REPORT")
    print("=" * 80)
    print()
    
    total = result['total_comparisons']
    
    # Header
    print(f"{'Stage':<8} {'Channels':<10} {'Spatial':<12} {'P':<6} {'Comparisons':<15} {'%':<8}")
    print("-" * 80)
    
    # Per-stage breakdown
    for stage, info in result['per_stage'].items():
        channels = info['channels']
        spatial = f"{info['spatial'][0]}×{info['spatial'][1]}"
        
        if stage == 0:
            p_str = "Adaptive"
        else:
            p_str = str(info['p_value'])
        
        comp = info['comparisons']
        percentage = (comp / total) * 100
        
        print(f"{stage:<8} {channels:<10} {spatial:<12} {p_str:<6} {comp:<15,} {percentage:>6.1f}%")
    
    print("-" * 80)
    print(f"{'TOTAL':<8} {'':<10} {'':<12} {'':<6} {total:<15,} {'100.0%':>6}")
    print("=" * 80)
    print()


def auto_generate_spatial_dims(initial_size, num_stages, downsample_after):
    """
    Automatically generate spatial dimensions based on downsampling.
    
    Args:
        initial_size: Tuple (H, W) for initial spatial size
        num_stages: Total number of stages
        downsample_after: List of stage indices after which to downsample by 2
    
    Returns:
        List of (H, W) tuples for each stage
    """
    spatial_dims = []
    H, W = initial_size
    
    for stage in range(num_stages):
        spatial_dims.append((H, W))
        
        # Check if we downsample after this stage
        if downsample_after and stage in downsample_after:
            H = math.ceil(H / 2)
            W = math.ceil(W / 2)
    
    return spatial_dims


# ============================================================================
# CONFIGURATION - Modify these parameters as needed
# ============================================================================

# Channel configuration (including input channel at index 0)
CHANNELS = [1, 37, 40, 60, 60, 80, 80, 100, 100]

# Initial spatial dimensions (H, W)
INITIAL_SIZE = (29, 18)

# Stage indices after which to downsample (0-indexed, e.g., [1, 3] means downsample after stage 1 and stage 3)
# This affects the spatial size for subsequent stages
DOWNSAMPLE_AFTER = [1, 3]

# P-decay values for stages 1 onwards (must have len = len(CHANNELS) - 1)
P_DECAY = [8, 6, 6, 4, 4, 4, 4, 4]

# Adaptive-P distribution for stage 0 (p_value: pixel_count)
ADAPTIVE_P_COUNTS = {4: 48, 6: 273, 8: 89, 10: 97, 12: 15}

PATTERNS = 2

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    
    # Generate spatial dimensions based on downsampling
    spatial_dims = auto_generate_spatial_dims(
        initial_size=INITIAL_SIZE,
        num_stages=len(CHANNELS),
        downsample_after=DOWNSAMPLE_AFTER
    )
    
    # Calculate comparisons
    result = calculate_comparisons_per_pattern(
        channels=CHANNELS,
        spatial_dims=spatial_dims,
        p_decay=P_DECAY,
        adaptive_p_counts=ADAPTIVE_P_COUNTS
    )
    
    # Print report
    print("\n" + "=" * 80)
    print(f"CONFIGURATION: {CHANNELS[1:]}")
    print("=" * 80)
    print_comparison_report(result)
    
    print(f"\nTotal comparisons per pattern: {result['total_comparisons']:,}")
    print(f"Total comparisons for all patterns (N={PATTERNS}): {result['total_comparisons'] * PATTERNS:,}")
    print()

import csv
from textgrid import TextGrid, IntervalTier
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)


# EXPORTS

def seg_to_textgrid(segment_list: List[Tuple[float, float, str]], tg_path: str, 
                    tier_name: str ="phones", fix_overlaps: bool =True) -> None:
    '''
    Convert a segment list to a TextGrid, the TextGrid format requires that intervals
    do not overlap
    '''
    # Sort and fix overlaps
    segment_list.sort(key=lambda x: x[0])
    cleaned_segments = []

    for i, (start, end, label) in enumerate(segment_list):
        if fix_overlaps:
            prev_end = cleaned_segments[-1][1] if i > 0 else 0
            if end <= start:
                logging.warning("The end of the segment has been found before the start of the segment, they will be swapped / altered")
                swp = end
                end = start + 0.001  # Ensure minimal duration
                start = swp
            if start < prev_end:
                start = prev_end  # Adjust start to prevent overlap
            if start < end:
                cleaned_segments.append([start, end, label])
    
    #print(cleaned_segments)
    # Build TextGrid
    tg = TextGrid()
    tier = IntervalTier(name=tier_name, minTime=cleaned_segments[0][0], maxTime=cleaned_segments[-1][1])

    for start, end, label in cleaned_segments:
        tier.add(start, end, label)

    tg.append(tier)
    tg.write(tg_path)
    print(f"✅ TextGrid saved to: {tg_path}")


def seg_to_csv(segment_list: List[Tuple[float, float, str]], csv_path: str) -> None:
    '''
    Convert a segment list to csv, disregarding overlaps or empty/swapped intervals
    '''
    with open(csv_path, mode="w", newline='') as file:
        writer = csv.writer(file)
        for interval in seg:
            writer.writerow(list(interval))
    print(f"✅ CSV saved to: {csv_path}")

# IMPORTS

def csv_to_seg(csv_path: str) -> List[Tuple[float, float, str]]:
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        seg = [
        (float(start), float(end), label) 
        for start, end, label in reader
        ]
        return seg


def textgrid_to_seg(tg_path: str, tier_name: str = "phones") -> List[Tuple[float, float, str]]:
    tg = TextGrid()
    tg.read(tg_path)
    
    try:
        tier = tg.getFirst(tier_name)
    except ValueError:
        raise ValueError(f"Tier '{tier_name}' not found in {tg_path}")

    segments = [
        (interval.minTime, interval.maxTime, interval.mark)
        for interval in tier.intervals
        if interval.mark.strip()  # Skip empty labels
    ]

    return segments

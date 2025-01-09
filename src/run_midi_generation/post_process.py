import pretty_midi
import os
import sys
import argparse
import traceback
import copy
import random
from collections import defaultdict
import math
import logging
import decimal
from itertools import groupby


# Set up logging
log_file = "output/logs/post_processing.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Ensure the log directory exists

logging.basicConfig(
    level=logging.DEBUG,  # Log all levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # Log to file, overwrite each run
        logging.StreamHandler(sys.stdout)         # Also log to console
    ]
)

# Define allowed transitions in functional harmony
allowed_transitions = {
    'I': ['IV', 'V', 'vi'],
    'ii': ['V', 'vii°'],
    'iii': ['vi', 'IV'],
    'IV': ['I', 'ii', 'V'],
    'V': ['I', 'vi'],
    'vi': ['ii', 'IV'],
    'vii°': ['I'],
    'i': ['iv', 'V'],
    'ii°': ['V', 'vii°'],
    'III': ['vi', 'IV'],
    'iv': ['V', 'i'],
    'V': ['i'],
    'VI': ['ii°', 'iv'],
    'vii°': ['i'],
    # Add more if needed
}


# Redirect stdout and stderr to the log file
class LoggerStream:
    def __init__(self, level):
        self.level = level
    
    def write(self, message):
        if message.strip():  # Avoid logging empty messages
            self.level(message)
    
    def flush(self):
        pass  # No-op

sys.stdout = LoggerStream(logging.info)
sys.stderr = LoggerStream(logging.error)


########################################################
# 1. KEY HANDLING & SCALE
########################################################

def parse_key_signature(key_str):
    key_str = key_str.strip().lower()
    parts = key_str.split()
    if len(parts) == 1:
        tonic = parts[0]
        mode = 'major'
    elif len(parts) == 2:
        tonic, mode = parts
    else:
        raise ValueError("Invalid key signature. e.g. 'C major', 'G# minor'")

    tonic = tonic.capitalize()
    mode = mode.lower()
    full_key = f"{tonic} {mode}"

    key_map = {
        # Majors
        "C major": 0,    "G major": 1,   "D major": 2,   "A major": 3,
        "E major": 4,    "B major": 5,   "F# major": 6,  "C# major": 7,
        "F major": -1,   "Bb major": -2, "Eb major": -3, "Ab major": -4,
        "Db major": -5,  "Gb major": -6, "Cb major": -7,
        # Minors
        "A minor": 0,    "E minor": 1,   "B minor": 2,   "F# minor": 3,
        "C# minor": 4,   "G# minor": 5,  "D# minor": 6,  "A# minor": 7,
        "D minor": -1,   "G minor": -2,  "C minor": -3,  "F minor": -4,
        "Bb minor": -5,  "Eb minor": -6, "Ab minor": -7
    }

    if full_key not in key_map:
        raise ValueError(f"Unrecognized key signature: '{full_key}'")
    return key_map[full_key]


def determine_tonic_mode(key_number):
    major_map = {
        0: ("C", "major"),   1: ("G", "major"),   2: ("D", "major"),   3: ("A", "major"),
        4: ("E", "major"),   5: ("B", "major"),   6: ("F#", "major"),  7: ("C#", "major"),
        -1: ("F", "major"),  -2: ("Bb", "major"), -3: ("Eb", "major"), -4: ("Ab", "major"),
        -5: ("Db", "major"), -6: ("Gb", "major"), -7: ("Cb", "major")
    }
    minor_map = {
        0: ("A", "minor"),   1: ("E", "minor"),   2: ("B", "minor"),   3: ("F#", "minor"),
        4: ("C#", "minor"),  5: ("G#", "minor"),  6: ("D#", "minor"),  7: ("A#", "minor"),
        -1: ("D", "minor"),  -2: ("G", "minor"),  -3: ("C", "minor"),  -4: ("F", "minor"),
        -5: ("Bb", "minor"), -6: ("Eb", "minor"), -7: ("Ab", "minor")
    }
    if key_number in major_map:
        return major_map[key_number]
    elif key_number in minor_map:
        return minor_map[key_number]
    else:
        # Fallback to C major if somehow out of range
        return ("C", "major")


def get_scale_pitches(tonic, mode):
    major_intervals = [0, 2, 4, 5, 7, 9, 11]
    minor_intervals = [0, 2, 3, 5, 7, 8, 10]
    note_to_pc = {
        'C': 0,
        'C#': 1, 'Db': 1,
        'D': 2,
        'D#': 3, 'Eb': 3,
        'E': 4, 'Fb': 4,
        'F': 5, 'E#': 5,
        'F#': 6, 'Gb': 6,
        'G': 7,
        'G#': 8, 'Ab': 8,
        'A': 9,
        'A#': 10, 'Bb': 10,
        'B': 11, 'Cb': 11, 'B#': 0
    }
    if tonic not in note_to_pc:
        raise ValueError(f"Invalid tonic note: {tonic}")

    tonic_pc = note_to_pc[tonic]
    intervals = major_intervals if mode == 'major' else minor_intervals
    scale_pcs = [(tonic_pc + i) % 12 for i in intervals]

    scale_pitches = set()
    for octave in range(11):  # 0..10
        for pc in scale_pcs:
            pitch = octave * 12 + pc
            if 0 <= pitch <= 127:
                scale_pitches.add(pitch)
    return scale_pitches


########################################################
# 2. LOW-LEVEL NOTE HELPERS
########################################################




def thin_notes(notes, max_density=10, window_size=0.5):
    sorted_notes = sorted(notes, key=lambda x: x.start)
    filtered = []
    i = 0
    while i < len(sorted_notes):
        w_start = sorted_notes[i].start
        w_end = w_start + window_size
        window_notes = []
        j = i
        while j < len(sorted_notes) and sorted_notes[j].start < w_end:
            window_notes.append(sorted_notes[j])
            j += 1
        if len(window_notes) > max_density:
            window_notes.sort(key=lambda n: n.velocity, reverse=True)
            filtered.extend(window_notes[:max_density])
        else:
            filtered.extend(window_notes)
        i += len(window_notes)
    return sorted(filtered, key=lambda x: x.start)


def remove_duplicate_notes(notes, time_threshold=0.02):
    sorted_notes = sorted(notes, key=lambda x: (x.start, x.pitch))
    out = []
    last = None
    for n in sorted_notes:
        if not last:
            out.append(n)
            last = n
        else:
            if abs(n.start - last.start) < time_threshold and n.pitch == last.pitch:
                # Extend the last note if n is longer
                if n.end > last.end:
                    last.end = n.end
            else:
                out.append(n)
                last = n
    return out


def condense_repeated_notes(notes, repeat_threshold=0.25):
    if not notes:
        return []
    pitch_dict = defaultdict(list)
    for note in notes:
        pitch_dict[note.pitch].append(note)

    condensed = []
    for pitch, group in pitch_dict.items():
        group_sorted = sorted(group, key=lambda x: x.start)
        prev = None
        for gn in group_sorted:
            if prev and (gn.start - prev.end) < repeat_threshold:
                # Merge
                if gn.end > prev.end:
                    prev.end = gn.end
            else:
                condensed.append(gn)
                prev = gn
    condensed.sort(key=lambda x: x.start)
    return condensed


def snap_notes_to_key(notes, scale_pitches):
    """
    Snap each note's pitch to the *nearest* pitch within scale_pitches,
    with a maximum shift of ±6 semitones. If none found, pick the absolutely closest pitch in the scale.
    """
    adjusted = []
    for note in notes:
        if note.pitch in scale_pitches:
            adjusted.append(note)
            continue

        candidates = [
            p for p in scale_pitches
            if (p >= note.pitch - 6) and (p <= note.pitch + 6)
        ]
        if not candidates:
            # fallback
            possible_pcs = sorted(scale_pitches)
            best = min(possible_pcs, key=lambda p: abs(p - note.pitch))
        else:
            best = min(candidates, key=lambda p: abs(p - note.pitch))
        
        new_note = copy.deepcopy(note)
        new_note.pitch = best
        adjusted.append(new_note)
    return adjusted


def adjust_velocity(notes, category):
    for note in notes:
        if category == "bass":
            note.velocity = min(int(note.velocity * 1.1), 127)
        elif category == "lead":
            note.velocity = max(int(note.velocity * 0.9), 20)
        elif category == "counter":
            note.velocity = max(int(note.velocity * 0.85), 18)
        elif category == "chords":
            note.velocity = max(int(note.velocity * 0.8), 18)
    return notes


########################################################
# 3. CHORD NAMING & ANALYSIS
########################################################

def chord_pitch_classes(notes):
    pcs = set(n.pitch % 12 for n in notes)
    return sorted(pcs)


def pc_to_note(pc):
    names = ["C", "C#", "D", "D#", "E", "F",
             "F#", "G", "G#", "A", "A#", "B"]
    return names[pc % 12]


def name_that_chord(pcset):
    """
    Simple chord detection: triads + some 7ths
    """
    triads = {
        (0, 4, 7): "maj",
        (0, 3, 7): "min",
        (0, 3, 6): "dim",
        (0, 4, 8): "aug"
    }
    sevenths = {
        (0, 4, 7, 10): "7",
        (0, 4, 7, 11): "maj7",
        (0, 3, 7, 10): "min7"
    }

    for inversion in range(len(pcset)):
        rotated = [((pcset[i] - pcset[inversion]) % 12) for i in range(len(pcset))]
        rotated_sorted = tuple(sorted(rotated))
        if rotated_sorted in triads:
            root_pc = pcset[inversion]
            quality = triads[rotated_sorted]
            return f"{pc_to_note(root_pc)}{quality}", root_pc, quality
        elif rotated_sorted in sevenths:
            root_pc = pcset[inversion]
            quality = sevenths[rotated_sorted]
            return f"{pc_to_note(root_pc)}{quality}", root_pc, quality

    return "Unknown", pcset[0] if pcset else 0, ""


def analyze_chord(notes):
    if not notes:
        return None
    notes_sorted = sorted(notes, key=lambda x: x.pitch)
    pcset = chord_pitch_classes(notes_sorted)
    chord_name, root_pc, quality = name_that_chord(pcset)
    bass_pc = notes_sorted[0].pitch % 12
    return {
        'chord_name': chord_name,
        'root_pc': root_pc,
        'quality': quality,
        'bass_pc': bass_pc,
    }


def analyze_chord_with_function(notes, key_signature):
    """
    Analyzes a chord and assigns a functional role based on the key signature.
    
    :param notes: List of pretty_midi.Note objects representing the chord.
    :param key_signature: String representing the key (e.g., 'C major').
    :return: Dictionary containing chord name, root, quality, functional role, start_time, and end_time.
    """
    chord_info = analyze_chord(notes)
    if not chord_info:
        return None
    
    # Parse key signature
    try:
        key_num = parse_key_signature(key_signature)
        tonic, mode = determine_tonic_mode(key_num)
    except ValueError:
        tonic, mode = ("C", "major")  # Default
    
    # Define scale degrees
    if mode == 'major':
        scale_degrees = {
            0: 'I',
            2: 'ii',
            4: 'iii',
            5: 'IV',
            7: 'V',
            9: 'vi',
            11: 'vii°'
        }
    else:
        scale_degrees = {
            9: 'i',
            11: 'ii°',
            0: 'III',
            2: 'iv',
            4: 'V',
            5: 'VI',
            7: 'vii°'
        }
    
    # Determine chord's root pitch class
    root_pc = chord_info['root_pc']
    
    # Calculate scale degree
    tonic_pc = parse_key_signature(key_signature)
    degree = (root_pc - tonic_pc) % 12
    functional_role = scale_degrees.get(degree, 'Unknown')
    
    # Determine start and end times
    start_time = min(note.start for note in notes)
    end_time = max(note.end for note in notes)
    
    chord_info['functional_role'] = functional_role
    chord_info['start_time'] = start_time
    chord_info['end_time'] = end_time
    
    return chord_info



def get_chord_pitches_from_role(role, key_signature):
    """
    Retrieves chord note names based on the functional role and key signature.
    
    :param role: Functional role (e.g., 'V', 'ii').
    :param key_signature: Key signature string (e.g., 'C major').
    :return: List of note names (e.g., ['G', 'B', 'D']).
    """
    key_num = parse_key_signature(key_signature)
    tonic, mode = determine_tonic_mode(key_num)
    
    if mode == 'major':
        functional_roles = {
            'I': ['C', 'E', 'G'],
            'ii': ['D', 'F', 'A'],
            'iii': ['E', 'G', 'B'],
            'IV': ['F', 'A', 'C'],
            'V': ['G', 'B', 'D'],
            'vi': ['A', 'C', 'E'],
            'vii°': ['B', 'D', 'F'],
        }
    else:
        functional_roles = {
            'i': ['A', 'C', 'E'],
            'ii°': ['B', 'D', 'F'],
            'III': ['C', 'E', 'G'],
            'iv': ['D', 'F', 'A'],
            'V': ['E', 'G#', 'B'],
            'VI': ['F', 'A', 'C'],
            'vii°': ['G#', 'B', 'D'],
        }
    
    chord_notes = functional_roles.get(role, ['C', 'E', 'G'])  # Default to C major if unknown
    return chord_notes

def note_name_to_midi(note_name, octave=4):
    """
    Converts a note name to a MIDI pitch number.
    
    :param note_name: String representing the note (e.g., 'C', 'G#').
    :param octave: Integer representing the octave.
    :return: MIDI pitch number.
    """
    note_to_pc = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'Fb': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7,
        'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11, 'B#': 0
    }
    pc = note_to_pc.get(note_name.upper(), 0)
    return pc + (octave + 1) * 12  # MIDI octave starts at -1


def check_and_correct_progression(chord_progression, key_signature):
    """
    Checks the chord progression for functional harmony and corrects invalid transitions.
    
    :param chord_progression: List of dictionaries containing chord info with functional roles.
    :param key_signature: Key signature string.
    :return: Corrected chord progression.
    """
    corrected_progression = [chord_progression[0]]  # Start with the first chord
    
    for i in range(1, len(chord_progression)):
        prev_chord = corrected_progression[-1]
        current_chord = chord_progression[i]
        
        allowed_next = allowed_transitions.get(prev_chord['functional_role'], [])
        if current_chord['functional_role'] in allowed_next:
            corrected_progression.append(current_chord)
        else:
            # Suggest a default transition, e.g., to V or I
            if 'V' in allowed_next:
                target_role = 'V'
            elif 'I' in allowed_next:
                target_role = 'I'
            else:
                target_role = allowed_next[0] if allowed_next else 'I'
            
            # Replace current chord's functional role
            chord_info = {
                'chord_name': target_role,
                'root_pc': (parse_key_signature(key_signature) + note_name_to_pc(get_chord_pitches_from_role(target_role, key_signature)[0])) % 12,
                'quality': 'major' if target_role.upper() in ['I', 'IV', 'V', 'III', 'VI'] else 'minor',
                'functional_role': target_role,
                'start_time': current_chord['start_time'],
                'end_time': current_chord['end_time']
            }
            corrected_progression.append(chord_info)
    
    return corrected_progression

def note_name_to_pc(note_name):
    """
    Converts a note name to its pitch class.
    
    :param note_name: String representing the note (e.g., 'C', 'G#').
    :return: Integer pitch class (0-11).
    """
    note_to_pc = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'Fb': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7,
        'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11, 'B#': 0
    }
    return note_to_pc.get(note_name.upper(), 0)


########################################################
# 4. ADVANCED VOICE LEADING
########################################################

def advanced_voice_leading(chord_groups, key_signature):
    """
    We reorder chord pitches from one chord to the next for minimal movement.
    Then we ensure we do NOT lose the root note if chord analysis can detect it.
    """
    if not chord_groups:
        return []
    
    # 1) Sort first chord by pitch
    chord_groups[0].sort(key=lambda x: x.pitch)
    new_groups = [chord_groups[0]]

    for i in range(1, len(chord_groups)):
        prev_group = new_groups[i - 1]
        curr_group = chord_groups[i]

        # We can do a quick chord analysis to find the root
        chord_info = analyze_chord_with_function(curr_group, key_signature)
        root_pc = chord_info['root_pc'] if chord_info else None

        prev_pitches = [n.pitch for n in prev_group]
        curr_pitches_original = sorted(n.pitch for n in curr_group)

        new_arr = []
        for v_idx, c_pitch in enumerate(curr_pitches_original):
            if v_idx < len(prev_pitches):
                # Voice lead: pick an octave near prev_pitches[v_idx]
                p_pitch = prev_pitches[v_idx]
                candidates = [c_pitch + (12 * o) for o in [-1, 0, 1]]
                chosen = min(candidates, key=lambda x: abs(x - p_pitch))
                new_arr.append(chosen)
            else:
                new_arr.append(c_pitch)

        # Sort final
        new_arr_sorted = sorted(new_arr)

        # Rebuild note objects
        new_chord = []
        original_sorted = sorted(curr_group, key=lambda x: x.pitch)
        for j, n in enumerate(original_sorted):
            nn = copy.deepcopy(n)
            nn.pitch = new_arr_sorted[j]
            new_chord.append(nn)

        # 2) **If chord has a known root, enforce that it’s in the chord** 
        # (i.e., if for some reason the root got dropped)
        if root_pc is not None:
            root_in_chord = any((note.pitch % 12) == root_pc for note in new_chord)
            if not root_in_chord and new_chord:
                # Insert a note at the chord’s average pitch but with root_pc
                avg_pitch = sum(nt.pitch for nt in new_chord) // len(new_chord)
                # Snap to the correct octave
                candidate_oct = avg_pitch // 12
                root_candidate = candidate_oct * 12 + root_pc

                # Choose the nearest candidate octave so it's not too far
                oct_variants = [root_candidate + 12*k for k in [-1,0,1]]
                best_root_pitch = min(oct_variants, key=lambda p: abs(p - avg_pitch))
                
                # Clone any note as a template
                template = new_chord[0]
                root_note = copy.deepcopy(template)
                root_note.pitch = best_root_pitch
                new_chord.append(root_note)

        new_groups.append(new_chord)

    return new_groups



########################################################
# 5. CHORD BLOCK PLACEMENT
########################################################

def get_fixed_chord_positions(total_bars):
    """
    For EXACT 8 bars: return measure 0,2,4,6
    Always guarantee 0 and 5 if total_bars >= 5
    """
    if total_bars < 2:
        return [0]

    # If total_bars == 8, explicitly return [0,2,4,6]
    if total_bars == 8:
        return [0, 2, 4, 6]
    
    # Otherwise, place chords every 2 bars, ensuring start and end
    positions = list(range(0, total_bars, 2))
    return [pos for pos in positions if pos < total_bars]


def place_chord_blocks_with_voice_leading(instrument, total_bars=8, tempo=120.0, time_threshold=0.2, key_signature="C major"):
    """
    Modified to make chords span full duration between chord changes,
    with the last chord going to the end of the piece.
    """
    notes = instrument.notes
    if not notes:
        logging.warning("No notes found in the instrument.")
        return
    
    # Basic timing calculations
    sec_per_beat = 60.0 / tempo
    bar_duration = 4 * sec_per_beat
    total_duration = total_bars * bar_duration
    
    # Group simultaneous notes into chords
    sorted_notes = sorted(notes, key=lambda x: x.start)
    chord_groups = []
    current_group = [sorted_notes[0]]
    current_pitches = {sorted_notes[0].pitch}
    
    for n in sorted_notes[1:]:
        if abs(n.start - current_group[0].start) <= time_threshold:
            if n.pitch not in current_pitches:
                current_group.append(n)
                current_pitches.add(n.pitch)
        else:
            if len(current_group) >= 2:
                chord_groups.append(current_group)
                logging.debug(f"Detected chord group starting at {current_group[0].start}: {[note.pitch for note in current_group]}")
            current_group = [n]
            current_pitches = {n.pitch}
    
    if len(current_group) >= 2:
        chord_groups.append(current_group)
        logging.debug(f"Detected final chord group starting at {current_group[0].start}: {[note.pitch for note in current_group]}")
    
    if not chord_groups:
        logging.warning("No chord groups detected. Ensure that chords have overlapping notes within the time_threshold.")
    
    # Voice leading
    chord_groups = advanced_voice_leading(chord_groups, key_signature)
    
    # Get fixed positions (always includes 0 and last bar)
    positions = [0]  # Always start at beginning
    if total_bars >= 2:
        positions.extend(range(2, total_bars, 2))
    
    logging.debug(f"Chord positions (bars): {positions}")
    
    # Place chords
    final_notes = []
    for i, chord_group in enumerate(chord_groups):
        if i >= len(positions):
            break
            
        start_bar = positions[i]
        # End at next position or total duration
        end_bar = positions[i + 1] if i + 1 < len(positions) else total_bars
        
        start_time = start_bar * bar_duration
        end_time = end_bar * bar_duration
        
        logging.debug(f"Placing chord {i+1} from {start_time} to {end_time}")
        
        # Place the chord
        for note in chord_group:
            new_note = copy.deepcopy(note)
            new_note.start = start_time
            new_note.end = end_time
            final_notes.append(new_note)
    
    # Make sure we have something at the end
    if not final_notes or final_notes[-1].end < total_duration - 0.1:
        if chord_groups:
            last_chord = chord_groups[-1]
            for note in last_chord:
                new_note = copy.deepcopy(note)
                new_note.start = final_notes[-1].end if final_notes else (total_duration - bar_duration)
                new_note.end = total_duration
                final_notes.append(new_note)
    
    # Replace the instrument's notes
    instrument.notes = sorted(final_notes, key=lambda x: x.start)
    
    logging.info(f"Total chords placed: {len(final_notes)}")


########################################################
# 6. CHORD COMPLETION
########################################################

def complete_two_note_chord(chord_notes, chord_info, scale_pitches):
    """
    Add a missing note to form a triad if it fits in scale.
    Minimally invasive approach.
    """
    root_pc = chord_info['root_pc']
    quality = chord_info['quality']

    intervals_map = {
        'maj': [0, 4, 7],
        'min': [0, 3, 7],
        'dim': [0, 3, 6],
        'aug': [0, 4, 8],
        '7':   [0, 4, 7, 10],
        'maj7':[0, 4, 7, 11],
        'min7':[0, 3, 7, 10]
    }
    intervals = intervals_map.get(quality, [0, 4, 7])  # default to major triad

    existing_pcs = {n.pitch % 12 for n in chord_notes}
    target_pcs = {(root_pc + i) % 12 for i in intervals}
    missing = target_pcs - existing_pcs

    if len(chord_notes) == 2 and missing:
        missing_pc = missing.pop()
        avg_pitch = sum(n.pitch for n in chord_notes) // len(chord_notes)
        base_oct = avg_pitch // 12
        new_pitch = base_oct * 12 + missing_pc
        
        # Snap to scale
        if new_pitch not in scale_pitches:
            new_pitch = min(scale_pitches, key=lambda p: abs(p - new_pitch))
        new_pitch = max(0, min(new_pitch, 127))

        new_note = pretty_midi.Note(
            velocity=chord_notes[0].velocity,
            pitch=new_pitch,
            start=chord_notes[0].start,
            end=chord_notes[0].end
        )
        chord_notes.append(new_note)
    
    return chord_notes


def basic_two_note_completion(chord_notes, scale_pitches):
    """
    If chord analysis fails, we guess a 3rd or 5th to add.
    """
    if len(chord_notes) < 2:
        return chord_notes

    chord_notes_sorted = sorted(chord_notes, key=lambda x: x.pitch)
    interval = chord_notes_sorted[1].pitch - chord_notes_sorted[0].pitch
    # Heuristic map
    guess_map = {
        3: 7,  # minor third -> add fifth
        4: 7,  # major third -> add fifth
        5: 4,  # perfect fourth -> add major third below
        7: 4,  # perfect fifth -> add major third below
    }
    offset = guess_map.get(interval % 12, 4)
    # Typically we add the note to the lower pitch to form a triad
    new_pitch = chord_notes_sorted[0].pitch + offset

    # Snap to scale
    if new_pitch not in scale_pitches:
        new_pitch = min(scale_pitches, key=lambda p: abs(p - new_pitch))
    new_pitch = max(0, min(new_pitch, 127))

    new_note = pretty_midi.Note(
        velocity=chord_notes_sorted[0].velocity,
        pitch=new_pitch,
        start=chord_notes_sorted[0].start,
        end=chord_notes_sorted[0].end
    )
    chord_notes.append(new_note)
    return chord_notes


########################################################
# 7. COUNTER / LEAD PATTERNS
########################################################

def create_alternating_pattern(
    instrument, 
    category, 
    scale_pitches, 
    num_bars, 
    tempo=120.0, 
    counter_hyperness=0.5
):
    """
    Create a pattern with noticeable difference in density based on counter_hyperness.
    - If hyperness is high (~1), produce more notes, bigger leaps, shorter durations.
    - If hyperness is low (~0), produce fewer notes, smaller leaps, longer durations.
    """
    if not instrument.notes:
        return
    
    sec_per_beat = 60.0 / tempo
    bar_duration = 4 * sec_per_beat
    total_duration = num_bars * bar_duration
    
    orig_pitches = sorted(set(note.pitch for note in instrument.notes))
    if not orig_pitches:
        return

    # Define hyperness effects
    max_leap = int(12 + 12 * counter_hyperness)  # up to 24 semitones if hyperness=1
    base_notes_per_bar = 1
    max_notes_per_bar = 5
    notes_per_bar = int(base_notes_per_bar + (max_notes_per_bar - base_notes_per_bar) * counter_hyperness)

    if notes_per_bar < 1:
        notes_per_bar = 1  # safety

    pattern_dur = bar_duration / float(notes_per_bar)

    new_notes = []
    current_time = 0.0
    last_pitch = random.choice(orig_pitches)

    while current_time < total_duration:
        # Find scale pitches within ±max_leap
        candidates = [p for p in scale_pitches if abs(p - last_pitch) <= max_leap]
        if not candidates:
            candidates = list(scale_pitches)

        # Avoid exact repetition if possible
        filtered = [c for c in candidates if c != last_pitch]
        if not filtered:
            filtered = candidates
        pitch = random.choice(filtered)

        # Duration logic
        if category == "lead":
            # Lead might hold notes a bit longer
            duration = pattern_dur * random.uniform(1.0, 1.5 - 0.3*counter_hyperness)
        else:
            # Counter: more hyperness => shorter durations
            duration = pattern_dur * (1.2 - 0.7*counter_hyperness)

        end_time = min(current_time + duration, total_duration)
        vel = random.randint(60, 100)
        new_note = pretty_midi.Note(
            velocity=vel,
            pitch=pitch,
            start=current_time,
            end=end_time
        )
        new_notes.append(new_note)

        last_pitch = pitch
        current_time += duration

    instrument.notes = sorted(new_notes, key=lambda x: x.start)


########################################################
# 8. RANGE & LEGATO
########################################################

def clamp_pitch(note, min_pitch, max_pitch):
    while note.pitch < min_pitch:
        note.pitch += 12
    while note.pitch > max_pitch:
        note.pitch -= 12
    return note


def apply_instrument_range(instrument, category):
    """
    Clamp pitch ranges based on category.
    """
    if category == "bass":
        min_p, max_p = 36, 48
    elif category == "chords":
        min_p, max_p = 52, 72
    elif category == "lead":
        min_p, max_p = 72, 84
    elif category == "counter":
        min_p, max_p = 60, 84
    else:
        min_p, max_p = 48, 84  # fallback
    
    for note in instrument.notes:
        clamp_pitch(note, min_p, max_p)


def make_legato(instrument, overlap=0.05):
    """
    Adjust note end times to create legato overlaps.
    """
    notes = sorted(instrument.notes, key=lambda x: x.start)
    if not notes:
        return
    chords = defaultdict(list)
    for note in notes:
        chords[note.start].append(note)
    start_times = sorted(chords.keys())

    for i in range(len(start_times)):
        curr_start = start_times[i]
        if i < len(start_times) - 1:
            next_start = start_times[i + 1]
            end_time = next_start - overlap
        else:
            end_time = curr_start + 4.0  # default duration for last chord

        for note in chords[curr_start]:
            note.end = max(note.start + 0.05, end_time)

    instrument.notes = sorted(notes, key=lambda x: x.start)


########################################################
# 9. OTHER PROCESSING
########################################################

def split_long_notes_with_variations(notes, tempo=120.0, max_duration_beats=2, total_duration=32):
    """
    Split long sustained notes into shorter notes with slight variations.
    """
    sec_per_beat = 60.0 / tempo
    max_duration = max_duration_beats * sec_per_beat
    new_notes = []
    
    for note in notes:
        duration = note.end - note.start
        if duration > max_duration:
            num_segments = max(2, int(duration / max_duration))
            segment_duration = duration / num_segments
            for i in range(num_segments):
                start_offset = random.uniform(-0.02, 0.02)
                dur_mod = random.uniform(0.95, 1.05)
                
                new_start = note.start + i * segment_duration + start_offset
                new_end = new_start + segment_duration * dur_mod
                new_start = max(note.start, new_start)
                new_end = min(note.end, new_end, total_duration)
                new_velocity = min(127, max(1, note.velocity + random.randint(-5,5)))

                new_note = pretty_midi.Note(
                    velocity=new_velocity,
                    pitch=note.pitch,
                    start=new_start,
                    end=new_end
                )
                new_notes.append(new_note)
        else:
            new_notes.append(note)
    
    return sorted(new_notes, key=lambda x: x.start)


def remove_note_overlaps(notes, keep="higher"):
    """
    If two notes overlap in time, keep the 'higher' or 'lower' pitched note only.
    """
    if keep == "higher":
        notes.sort(key=lambda n: (n.start, -n.pitch))
    else:
        notes.sort(key=lambda n: (n.start, n.pitch))

    filtered = []
    for note in notes:
        if not filtered:
            filtered.append(note)
            continue
        last = filtered[-1]
        if note.start < last.end:  # overlap
            if keep == "higher":
                if note.pitch > last.pitch:
                    filtered[-1] = note
            else:
                if note.pitch < last.pitch:
                    filtered[-1] = note
        else:
            filtered.append(note)
    return filtered


########################################################
# 10. OPTIMIZE MIDI
########################################################

def optimize_midi(midi):
    """
    Remove same-pitch overlaps within each instrument.
    """
    for instr in midi.instruments:
        instr.notes.sort(key=lambda n: (n.pitch, n.start))
        out = []
        last = None
        for n in instr.notes:
            if last and (n.pitch == last.pitch) and (n.start < last.end):
                if n.end > last.end:
                    last.end = n.end
            else:
                out.append(n)
                last = n
        instr.notes = out
    return midi


########################################################
# 11. SPLIT & ASSIGN
########################################################

def split_midi_by_instrument(midi):
    """
    Split MIDI into a dictionary of sub-MIDIs, each containing one instrument.
    """
    out_dict = {}
    for i, instr in enumerate(midi.instruments):
        sub = pretty_midi.PrettyMIDI()
        sub.time_signature_changes = copy.deepcopy(midi.time_signature_changes)
        sub.key_signature_changes = copy.deepcopy(midi.key_signature_changes)

        new_instr = pretty_midi.Instrument(
            program=instr.program,
            is_drum=instr.is_drum,
            name=instr.name or f"Instrument_{i}"
        )
        new_instr.notes = copy.deepcopy(instr.notes)
        sub.instruments.append(new_instr)

        name_key = instr.name.strip() if instr.name and instr.name.strip() else f"Instrument_{i}"
        out_dict[name_key] = sub
    return out_dict
def get_bar_duration(midi_obj):
    """
    Get the duration of one bar based on the MIDI's tempo.
    Returns duration in seconds.
    """
    tempo_times, tempos = midi_obj.get_tempo_changes()
    first_tempo = tempos[0] if tempos.size > 0 else 120.0
    return 4 * (60.0 / first_tempo)  # 4 beats per bar * seconds per beat

def trim_midi_by_bars(midi_obj, num_bars, default_bpb=4):
    """
    Strictly trim the MIDI to exactly num_bars length.
    Any notes extending beyond this are either trimmed or removed.
    """
    if not midi_obj or not midi_obj.instruments:
        return midi_obj
        
    bar_duration = get_bar_duration(midi_obj)
    total_duration = num_bars * bar_duration
    
    for instrument in midi_obj.instruments:
        # Remove any notes that start after the cutoff
        instrument.notes = [note for note in instrument.notes if note.start < total_duration]
        
        # Trim any notes that extend beyond the cutoff
        for note in instrument.notes:
            if note.end > total_duration:
                note.end = total_duration
            # Ensure no zero-length notes
            if note.end <= note.start:
                note.end = min(note.start + 0.1, total_duration)
    
    # Also trim time signature and key signature changes
    midi_obj.time_signature_changes = [
        ts for ts in midi_obj.time_signature_changes 
        if ts.time < total_duration
    ]
    
    midi_obj.key_signature_changes = [
        ks for ks in midi_obj.key_signature_changes 
        if ks.time < total_duration
    ]
    
    return midi_obj
def assign_instruments_to_categories(split_dict):
    """
    Assign instruments to categories: chords, lead, counter, bass.
    """
    cat_list = ["chords", "lead", "counter", "bass"]
    free_cats = set(cat_list)
    assigned = {}

    for name, sub_midi in split_dict.items():
        if not free_cats:
            assigned[name] = None
            continue

        instr = sub_midi.instruments[0]
        n_lower = (instr.name or "").lower()
        pitches = [nt.pitch for nt in instr.notes]
        if not pitches:
            assigned[name] = None
            continue
        maxp = max(pitches)

        # Heuristic:
        if "bass" in n_lower and "bass" in free_cats:
            assigned[name] = "bass"
            free_cats.remove("bass")
            logging.info(f"Assigned instrument '{name}' to category 'bass'.")
        elif maxp < 60 and "bass" in free_cats:
            assigned[name] = "bass"
            free_cats.remove("bass")
            logging.info(f"Assigned instrument '{name}' to category 'bass' (pitch-based).")
        else:
            # Check for chord clusters
            chord_map = defaultdict(list)
            for nt in instr.notes:
                st_key = round(nt.start, 2)
                chord_map[st_key].append(nt)
            chord_groups = [grp for grp in chord_map.values() if len(grp) >= 2]

            # If it forms a decent chunk of chords, call it 'chords'
            if len(chord_groups) >= 2 and "chords" in free_cats:
                assigned[name] = "chords"
                free_cats.remove("chords")
                logging.info(f"Assigned instrument '{name}' to category 'chords'.")
            else:
                # Next preference lead -> counter
                if "lead" in free_cats:
                    assigned[name] = "lead"
                    free_cats.remove("lead")
                    logging.info(f"Assigned instrument '{name}' to category 'lead'.")
                elif "counter" in free_cats:
                    assigned[name] = "counter"
                    free_cats.remove("counter")
                    logging.info(f"Assigned instrument '{name}' to category 'counter'.")
                else:
                    assigned[name] = None
                    logging.warning(f"No category assigned for instrument '{name}'.")

    leftover = list(free_cats)
    none_insts = [k for k, v in assigned.items() if v is None]
    for i, inst_name in enumerate(none_insts):
        if i < len(leftover):
            assigned[inst_name] = leftover[i]
            logging.info(f"Assigned leftover category '{leftover[i]}' to instrument '{inst_name}'.")
        else:
            assigned[inst_name] = None
            logging.warning(f"No categories left for instrument '{inst_name}'.")
    return assigned


########################################################
# 12. GAP FILL (Optional)
########################################################

def enhanced_fill_midi_gaps(
    midi_dict, 
    cat_map, 
    key_signature, 
    categories=["counter"], 
    num_bars=8, 
    counter_hyperness=0.5
):
    """
    If a 'counter' is extremely sparse, fill it with an alternating pattern before final processing.
    """
    try:
        key_num = parse_key_signature(key_signature)
        tonic, mode = determine_tonic_mode(key_num)
        scale_pitches = get_scale_pitches(tonic, mode)
    except ValueError as ve:
        logging.error(f"Error parsing key signature: {ve}")
        return

    for name, sub_midi in midi_dict.items():
        category = cat_map.get(name)
        if category in categories and sub_midi.instruments:
            instr = sub_midi.instruments[0]
            tempo_times, tempos = sub_midi.get_tempo_changes()
            tempo = tempos[0] if tempos else 120.0

            # If too few notes, create pattern
            notes_per_bar = len(instr.notes) / float(num_bars)
            if notes_per_bar < 1.0:
                logging.info(f"Filling gaps for category '{category}' in '{name}', hyperness={counter_hyperness}.")
                create_alternating_pattern(
                    instr, 
                    category, 
                    scale_pitches,
                    num_bars,
                    tempo=tempo,
                    counter_hyperness=counter_hyperness
                )


def handle_overlapping_start_notes(midi_dict, cat_map, categories, num_bars=8):
    """
    OPTIONAL function: ensure not too many instruments start notes at the exact same moment.
    Moves any "excess" starting notes after the first bar.
    """
    # Collect instruments by category
    category_instruments = []
    for name, sub_midi in midi_dict.items():
        category = cat_map.get(name)
        if category in categories:
            category_instruments.append(sub_midi.instruments[0])

    if len(category_instruments) < 2:
        logging.info("No overlapping instruments to handle.")
        return

    # Basic tempo
    first_midi = next(iter(midi_dict.values()))
    tempo_times, tempos = first_midi.get_tempo_changes()
    tempo = tempos[0] if tempos else 120.0
    bar_duration = 4 * (60.0 / tempo)
    total_duration = num_bars * bar_duration
    start_threshold = bar_duration  # the first bar

    # Collect all notes that start within the first bar
    overlapping_notes = []
    for instr in category_instruments:
        for note in instr.notes:
            if note.start < start_threshold:
                overlapping_notes.append((instr, note))

    # If more than 2, shift them
    if len(overlapping_notes) > 2:
        logging.info(f"Found {len(overlapping_notes)} overlapping start notes. Shifting extras.")
        for i, (instr, note) in enumerate(overlapping_notes):
            if i >= 2:  # Keep only two in the first bar
                new_start = (total_duration / 2.0) + 0.5
                shift_amount = new_start - note.start
                note.start += shift_amount
                note.end += shift_amount

                # Ensure notes don't exceed total_duration
                if note.end > total_duration:
                    note.end = total_duration
                if note.start > total_duration:
                    note.start = total_duration - 0.1
                    note.end = total_duration
    else:
        logging.info("No excessive overlapping starting notes found.")


########################################################
# 13. CATEGORY-SPECIFIC PROCESSING
########################################################


# First fix: Ensure chords consistently have 3 notes by improving the chord completion logic



def process_chords_advanced_with_functional_harmony(instrument, key_signature, num_bars=8, tempo=120.0):
    """
    Advanced processing for chord instruments with functional harmony enforcement.
    """
    logging.info("Processing chords with functional harmony.")
    
    try:
        key_num = parse_key_signature(key_signature)
        tonic, mode = determine_tonic_mode(key_num)
        scale_pitches = get_scale_pitches(tonic, mode)
    except ValueError as ve:
        logging.error(f"Error parsing key signature: {ve}")
        tonic, mode = ("C", "major")  # Default to C major
        scale_pitches = get_scale_pitches(tonic, mode)
    
    # Snap notes to key
    instrument.notes = snap_notes_to_key(instrument.notes, scale_pitches)
    
    # Place chord blocks (existing functionality)
    place_chord_blocks_with_voice_leading(
        instrument,
        total_bars=num_bars,
        tempo=tempo,
        time_threshold=0.05,
        key_signature=key_signature
    )
    
    # Group notes by their start times to form chords
    sorted_notes = sorted(instrument.notes, key=lambda x: x.start)
    grouped_notes = [list(group) for _, group in groupby(sorted_notes, key=lambda x: round(x.start, 2))]
    
    # Analyze each chord group
    chord_progression = []
    for chord_group in grouped_notes:
        chord_info = analyze_chord_with_function(chord_group, key_signature)
        if chord_info:
            chord_progression.append(chord_info)
    
    # Check and correct progression
    corrected_progression = check_and_correct_progression(chord_progression, key_signature)
    
    # Reconstruct notes based on corrected progression
    final_notes = []
    for chord_info in corrected_progression:
        # Get chord note names based on functional role
        chord_note_names = get_chord_pitches_from_role(chord_info['functional_role'], key_signature)
        # Convert note names to MIDI pitches
        midi_pitches = [note_name_to_midi(note, octave=4) for note in chord_note_names]
        
        # Create new notes
        for pitch in midi_pitches:
            new_note = pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=chord_info['start_time'],
                end=chord_info['end_time']
            )
            final_notes.append(new_note)
    
    # Replace instrument's notes with corrected chords
    instrument.notes = sorted(final_notes, key=lambda x: x.start)
    
    # Apply additional processing
    make_legato(instrument, overlap=0.05)
    
    # Ensure all notes are within total duration
    bar_duration = 4 * (60.0 / tempo)
    total_duration = num_bars * bar_duration
    for note in instrument.notes:
        if note.end > total_duration:
            note.end = total_duration
        if note.start > total_duration:
            note.start = total_duration - 0.1
            note.end = total_duration
    
    # Snap to key again after corrections
    instrument.notes = snap_notes_to_key(instrument.notes, scale_pitches)
    
    logging.info("Chords processing with functional harmony completed.")

# Second fix: Improve quantization to match FL Studio's behavior

def process_lead_or_counter_advanced_enhanced(instrument, key_signature, category, num_bars=8, counter_hyperness=0.5):
    """
    Enhanced processing for lead and counter parts:
      - Snap to key
      - Remove overlaps
      - Split long notes
      - If sparse, create alternating pattern
      - Clamp pitch ranges
      - Apply legato
    """
    try:
        key_num = parse_key_signature(key_signature)
        tonic, mode = determine_tonic_mode(key_num)
        scale_pitches = get_scale_pitches(tonic, mode)
    except ValueError as ve:
        logging.error(f"Error parsing key signature: {ve}")
        return
    
    instrument.notes = snap_notes_to_key(instrument.notes, scale_pitches)
    instrument.notes = remove_note_overlaps(instrument.notes, keep="higher" if category=="lead" else "lower")
    
    # Calculate total_duration
    tempo = 120.0  # you can refine by reading actual tempo from the file
    sec_per_beat = 60.0 / tempo
    bar_duration = 4 * sec_per_beat
    total_duration = num_bars * bar_duration

    instrument.notes = split_long_notes_with_variations(
        instrument.notes,
        tempo=tempo,
        max_duration_beats=2,
        total_duration=total_duration
    )
    
    # Check sparseness
    notes_per_bar = len(instrument.notes) / float(num_bars)
    if notes_per_bar < 1.0:
        logging.info(f"Instrument '{instrument.name}' is sparse. Creating pattern for '{category}', hyperness={counter_hyperness}.")
        create_alternating_pattern(
            instrument,
            category,
            scale_pitches,
            num_bars,
            tempo=tempo,
            counter_hyperness=counter_hyperness
        )

    apply_instrument_range(instrument, category)
    make_legato(instrument, overlap=0.03 if category == "counter" else 0.05)
    
    # Final cleanup
    instrument.notes = remove_note_overlaps(instrument.notes, keep="higher" if category=="lead" else "lower")
    instrument.notes = snap_notes_to_key(instrument.notes, scale_pitches)
    instrument.notes = remove_excessive_repeats(instrument.notes, max_repeats=2)




def quantize_to_half_beat_in_fl_style(midi_obj, notes, num_bars=8, preserve_duration=True):
    """
    Snap notes to the nearest half-beat in a given MIDI object.
    """
    ppq = midi_obj.resolution  # Ticks per quarter note
    ticks_per_half_beat = ppq // 2
    ticks_per_bar = 4 * ppq
    total_ticks = ticks_per_bar * num_bars

    # Generate valid tick positions for snapping
    valid_ticks = [i * ticks_per_half_beat for i in range((total_ticks // ticks_per_half_beat) + 1)]

    quantized_notes = []
    for note in notes:
        # Convert start and end times to ticks
        start_ticks = round(midi_obj.time_to_tick(note.start))
        end_ticks = round(midi_obj.time_to_tick(note.end))

        # Find nearest valid tick for the start time
        nearest_start_tick = min(valid_ticks, key=lambda x: abs(x - start_ticks))
        
        if preserve_duration:
            # Preserve duration but ensure end does not exceed total_ticks
            duration_ticks = end_ticks - start_ticks
            nearest_end_tick = nearest_start_tick + duration_ticks
            nearest_end_tick = min(nearest_end_tick, total_ticks)
        else:
            # Quantize end time to the nearest valid tick
            nearest_end_tick = min(valid_ticks, key=lambda x: abs(x - end_ticks))
            nearest_end_tick = min(nearest_end_tick, total_ticks)

        # Ensure minimum duration of one half-beat
        if nearest_end_tick <= nearest_start_tick:
            nearest_end_tick = min(nearest_start_tick + ticks_per_half_beat, total_ticks)

        # Convert ticks back to time
        new_note = pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=midi_obj.tick_to_time(nearest_start_tick),
            end=midi_obj.tick_to_time(nearest_end_tick)
        )
        quantized_notes.append(new_note)

    return sorted(quantized_notes, key=lambda x: x.start)


def post_process_sub_midi(sub_midi, category, key_signature, num_bars=None, counter_hyperness=0.5):
    """
    Master function for final processing of one sub-MIDI/instrument.
    Includes functional harmony enforcement.
    """
    if not sub_midi.instruments:
        logging.warning("No instruments in sub_midi.")
        return sub_midi
    
    instr = sub_midi.instruments[0]
    try:
        tempo_times, tempos = sub_midi.get_tempo_changes()
    except Exception as e:
        tempos = []
        logging.error(f"Error reading tempo: {e}")
    first_tempo = tempos[0] if tempos.size > 0 else 120.0

    # Basic cleanup
    instr.notes = thin_notes(instr.notes, max_density=12, window_size=0.5)
    instr.notes = remove_duplicate_notes(instr.notes, time_threshold=0.02)
    instr.notes = condense_repeated_notes(instr.notes, repeat_threshold=0.25)

    # Snap to key
    try:
        key_num = parse_key_signature(key_signature)
        tonic, mode = determine_tonic_mode(key_num)
        scale_pitches = get_scale_pitches(tonic, mode)
    except ValueError as ve:
        logging.error(f"Error parsing key signature: {ve}")
        return sub_midi
    instr.notes = snap_notes_to_key(instr.notes, scale_pitches)

    # Adjust velocity
    instr.notes = adjust_velocity(instr.notes, category if category else "")
    
    bars = num_bars if num_bars else 8

    # Category-specific processing
    if category == "chords":
        # Process chords with functional harmony
        process_chords_advanced_with_functional_harmony(instr, key_signature, num_bars=bars, tempo=first_tempo)
        apply_instrument_range(instr, category)
    elif category in ["lead", "counter", "bass"]:
        process_lead_or_counter_advanced_enhanced(
            instr, key_signature, category, 
            num_bars=bars, 
            counter_hyperness=counter_hyperness
        )
    else:
        # Fallback
        apply_instrument_range(instr, None)
        instr.notes = snap_notes_to_key(instr.notes, scale_pitches)

    # Final snap to key
    instr.notes = snap_notes_to_key(instr.notes, scale_pitches)
    optimize_midi(sub_midi)

    # Clamp all notes to total duration
    bar_duration = 4 * (60.0 / first_tempo)
    total_duration = bars * bar_duration
    for note in instr.notes:
        if note.end > total_duration:
            note.end = total_duration
        if note.start > total_duration:
            note.start = total_duration - 0.1
            note.end = total_duration

    # Assign the final key signature
    sub_midi.key_signature_changes = [pretty_midi.KeySignature(key_num, 0)]

    # Final sanity check
    for note in instr.notes:
        if note.pitch not in scale_pitches:
            logging.error(f"Note {note.pitch} is not in the key after final snap.")

    # Quantize to half-beat
    instr.notes = quantize_to_half_beat_in_fl_style(sub_midi, instr.notes, num_bars=bars, preserve_duration=True)

    logging.info(f"Finished processing instrument '{instr.name}' as '{category}'.")
    return sub_midi


########################################################
# 14. FINAL POST PROCESS
########################################################

def remove_excessive_repeats(notes, max_repeats=2):
    """
    Remove excessive repeated notes, keeping only max_repeats consecutive repetitions.
    """
    if not notes:
        return []
    
    result = []
    current_pitch = None
    repeat_count = 0
    
    for note in sorted(notes, key=lambda x: x.start):
        if note.pitch == current_pitch:
            repeat_count += 1
            if repeat_count <= max_repeats:
                result.append(note)
        else:
            current_pitch = note.pitch
            repeat_count = 1
            result.append(note)
            
    return result




########################################################
# 15. MAIN
########################################################

def main():
    random.seed(42)  # For reproducible randomness

    parser = argparse.ArgumentParser(description="Enhanced MIDI post-processor with precise chord placement and quantization.")
    parser.add_argument("--input", required=True, help="Input MIDI file.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--key", default="C major", help="Key signature to snap everything.")
    parser.add_argument("--num_bars", type=int, default=8,
                        help="Trim or ensure each sub MIDI to that many bars.")
    parser.add_argument("--counter_hyperness", type=float, default=0.5,
                        help="For counters (0=least active, 1=most active).")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        logging.error(f"Error: '{args.input}' does not exist.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        midi = pretty_midi.PrettyMIDI(args.input)
        split_dict = split_midi_by_instrument(midi)
        cat_map = assign_instruments_to_categories(split_dict)

        # Handle overlapping starts if desired
        handle_overlapping_start_notes(split_dict, cat_map, categories=["lead", "counter"], num_bars=args.num_bars)

        # Trim each sub MIDI
        if args.num_bars:
            for name, sub_midi in split_dict.items():
                split_dict[name] = trim_midi_by_bars(sub_midi, args.num_bars)

        # Fill gaps for counters
        enhanced_fill_midi_gaps(
            split_dict, 
            cat_map, 
            args.key,
            categories=["counter"], 
            num_bars=args.num_bars,
            counter_hyperness=args.counter_hyperness
        )

        # Post-process each instrument
        for name, sub_midi in split_dict.items():
            cat = cat_map.get(name)
            if cat:
                logging.info(f"Post-processing '{name}' as '{cat}'.")
                processed = post_process_sub_midi(
                    sub_midi, 
                    cat,
                    args.key,
                    num_bars=args.num_bars,
                    counter_hyperness=args.counter_hyperness
                )
                # Save
                safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '_', '-')]).rstrip()
                if not safe_name:
                    safe_name = "Unnamed"
                out_file = os.path.join(args.output_dir, f"{cat}_{safe_name}.mid")
                processed.write(out_file)
                logging.info(f"Wrote {cat} -> {out_file}")
            else:
                logging.warning(f"No category assigned for '{name}'. Skipping save or optionally save raw.")

        logging.info("MIDI post-processing completed successfully.")
    
    except AssertionError as ae:
        logging.error(f"Assertion Error: {ae}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()  
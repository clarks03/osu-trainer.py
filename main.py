from pydub import AudioSegment
# from pydub.effects import speedup
import librosa, numpy as np
import soundfile as sf
from audiostretchy.stretch import stretch_audio
import os
import ffmpy

def convert_file_to_dict(file_name: str) -> dict:
    """Converts the contents of <file_name>
    into a dictionary, where each key represents a
    section of the .osu file, and each value either represents
    a dictionary (for categories that use key-value pairs),
    or an array (for categories that use comma-separated lines.
    """

    # Assume the file is already open
    
    # Each section is split into blank line-separated sections.
    # Meaning, new sections are separated by one blank line.

    with open(file_name, "r") as file:
        # In case we want to do anything with the file format version
        file_format_version = file.readline()


        is_in_section = False
        osu_file_dict = {}
        current_header = ""
        for line in file:
            line = line.strip()
            # Ignore comments
            if line.startswith("//"):
                continue
            # Checking for line break, aka in between sections
            if line == "":
                is_in_section = False
            else:
                # This checks for the first line after a section break.
                # THis means that this is the header for the section.
                if not is_in_section:
                    is_in_section = True
                    current_header = line
                    # Depending on what the current header is, we may want a list
                    # instead of a dict to store the values.

                    # e.g. difficulty = dict, hitobjects = list 
                    if current_header == "[Events]" or current_header == "[TimingPoints]" or current_header == "[HitObjects]":
                        osu_file_dict[current_header] = []
                    else:
                        osu_file_dict[current_header] = {}
                # Standard line within the section
                else:
                    # Most of the headers have different behaviour for its values.
                    # e.g. General has key: value pairs, whereas metadata has key:value pairs.
                    if current_header == "[General]" or current_header == "[Editor]":
                        line = line.split(": ")
                        osu_file_dict[current_header][line[0]] = line[1]
                    elif current_header == "[Metadata]" or current_header == "[Difficulty]":
                        line = line.split(":")
                        osu_file_dict[current_header][line[0]] = line[1]
                    elif current_header == "[Colours]":
                        line = line.split(" : ")
                        osu_file_dict[current_header][line[0]] = line[1]
                    else:
                        line = line.split(",")
                        osu_file_dict[current_header].append(line)


        return osu_file_dict


def speedup_osu_file(d: dict, rate: float) -> dict:
    """Speeds up an osu! file (converted to a dictionary)
    by <rate> times.
    Also handles audio speedup in this stage.
    """
    # Setting the difficulty name up here.
    # I'm doing this because the rate might potentially be decreased in future steps,
    # and I want to keep the diffnames consistent.
    d["[Metadata]"]["Version"] = f"{d['[Metadata]']['Version']} {rate}x"

    # Before we do anything, we're going to simulate generating the approach rate and 
    # overall difficulty.
    # If either of them are >10, we will divide the rate by 1.5,
    # and scale it up later (using the double time mod).

    # I'm not going to document the calculations here;
    # it is better documented below (when I actually set the approach rate &
    # overall difficulty).
    orig_rate = rate
    approach_rate = int(d["[Difficulty]"]["ApproachRate"])
    if approach_rate < 5:
        preempt = 1200 + 600 * (5 - approach_rate) / 5
    elif approach_rate > 5:
        preempt = 1200 - 750 * (approach_rate - 5) / 5
    else:
        preempt = 1200
    preempt /= rate

    if preempt > 1200:
        new_approach_rate = (preempt - 1800) / -120
    elif preempt < 1200:
        new_approach_rate = (preempt - 1950) / -150
    else:
        new_approach_rate = 5
    overall_difficulty = int(d["[Difficulty]"]["OverallDifficulty"])
    hit_window = 80 - 6 * overall_difficulty
    hit_window /= rate
    new_overall_difficulty = (hit_window - 80) / -6

    if new_approach_rate > 10 or new_overall_difficulty > 10:
        rate /= 1.5

    # General adjustments
    d["[General]"]["AudioLeadIn"] = round(int(d["[General]"]["AudioLeadIn"]) / rate)
    d["[General]"]["PreviewTime"] = round(int(d["[General]"]["PreviewTime"]) / rate)

    # Editor Adjustments
    # editor_bookmarks_adjusted = [int(i) for i in d["[Editor]"]["Bookmarks"]]
    if "Bookmarks" in d["[Editor]"]:
        editor_bookmarks = d["[Editor]"]["Bookmarks"].split(",")
        editor_bookmarks_adjusted = []
        
        for i in range(len(editor_bookmarks)):
            editor_bookmarks_adjusted.append(round(int(editor_bookmarks[i]) / rate))

        d["[Editor]"]["Bookmarks"] = ",".join([str(i) for i in editor_bookmarks_adjusted])


    # Difficulty Adjustments

    # APPROACH RATE
    # Step 1: determine the approach rate
    approach_rate = int(d["[Difficulty]"]["ApproachRate"])

    # Step 2: calculate the preempt given the approach rate
    if approach_rate < 5:
        preempt = 1200 + 600 * (5 - approach_rate) / 5
    elif approach_rate > 5:
        preempt = 1200 - 750 * (approach_rate - 5) / 5
    else:
        preempt = 1200

    # Step 3: adjust the preempt depending on the rate
    preempt /= rate

    # Step 4: Reverse engineer the new approach rate given the new preempt
    if preempt > 1200:
        new_approach_rate = (preempt - 1800) / -120
    elif preempt < 1200:
        new_approach_rate = (preempt - 1950) / -150
    else:
        new_approach_rate = 5

    # Step 5: set the approach rate (with 1 decimal point precision)
    # Capping at 10 for now (might do some DT shenanigans down the line)
    d["[Difficulty]"]["ApproachRate"] = min(round(new_approach_rate, 1), 10)

    # OVERALL DIFFICULTY
    # Step 1: Determine the overall difficulty
    overall_difficulty = int(d["[Difficulty]"]["OverallDifficulty"])

    # Step 2: calculate the hit window
    hit_window = 80 - 6 * overall_difficulty

    # Step 3: scale down the hit window according to the rate
    hit_window /= rate

    # Step 4: Reverse engineer the new overall difficulty
    new_overall_difficulty = (hit_window - 80) / -6

    # Step 5: set the overall difficulty (with 1 decimal point precision)
    # Capping at 10 for now (might do some DT shenanigans down the line)
    d["[Difficulty]"]["OverallDifficulty"] = min(round(new_overall_difficulty, 1), 10)

    # Events Adjustments
    events = d["[Events]"]
    events_adjusted = []

    for event in events:
        # Event is an array. The second element (index 1) is startTime.
        new_event = event.copy()
        print(new_event)
        new_event[1] = round(int(event[1]) / rate)

        # Breaks have an endTime as well.
        if event[0] == 2 or event[0] == "Break":
            new_event[2] = round(int(event[2]) / rate)

        events_adjusted.append(new_event)

    d["[Events]"] = events_adjusted

    # TimingPoints Adjustments
    timing_points = d["[TimingPoints]"]
    timing_points_adjusted = []

    for timing_point in timing_points:
        new_timing_point = timing_point.copy()
        # timing_point[0] is time, needs to be adjusted
        new_timing_point[0] = round(int(timing_point[0]) / rate)

        # If the timing point is uninherited (timing_point[6] == 1), timing_point[1] also needs to be
        # adjusted.
        if timing_point[6] == "1":
            new_timing_point[1] = float(timing_point[1]) / rate

        timing_points_adjusted.append(new_timing_point)

    d["[TimingPoints]"] = timing_points_adjusted

    # Colours Adjustments
    # N/A

    # HitObjects Adjustments
    # This is the big tuna. The big kahuna.
    hit_objects = d["[HitObjects]"]
    hit_objects_adjusted = []

    for hit_object in hit_objects:
        new_hit_object = hit_object.copy()

        # hit_object[2] is always time, and needs to be adjusted.
        new_hit_object[2] = round(int(hit_object[2]) / rate)

        # Additionally, if the type of object is a spinner, it has an end time.
        if int(hit_object[3]) & 0b00001000 != 0:
            new_hit_object[5] = round(int(hit_object[5]) / rate)

        hit_objects_adjusted.append(new_hit_object)

    d["[HitObjects]"] = hit_objects_adjusted


    # Also speeding up the audio in this step
    # Using the <orig_rate> variable here because I want to keep this name
    # consistent with the (potentially scaled down) rate.

    # I'm going to need the name of the audio file without the .mp3, 
    # because I'm going to replace it with a .wav file.

    filename_no_path = d["[General]"]["AudioFilename"][:d["[General]"]["AudioFilename"].rfind(".")]
    speedup_audio_file(d["[General]"]["AudioFilename"], rate, f"{orig_rate}-{filename_no_path}.mp3")
    d["[General]"]["AudioFilename"] = f"{orig_rate}-{filename_no_path}.mp3"

    return d


def convert_dict_to_file(d: dict, outfile_path: str):
    """Outputs a dictionary <d> corresponding to a sped-up osu! file
    to a file location specified by <outfile_path>.
    """
    with open(outfile_path, "w") as file:
        file.write("osu file format v14\n")
        file.write("\n")

        # [General]
        file.write("[General]\n")
        for key, value in d["[General]"].items():
            file.write(f"{key}: {value}\n")
        file.write("\n")

        # [Editor]
        file.write("[Editor]\n")
        for key, value in d["[Editor]"].items():
            file.write(f"{key}: {value}\n")
        file.write("\n")

        # [Metadata]
        file.write("[Metadata]\n")
        for key, value in d["[Metadata]"].items():
            file.write(f"{key}:{value}\n")
        file.write("\n")

        # [Difficulty]
        file.write("[Difficulty]\n")
        for key, value in d["[Difficulty]"].items():
            file.write(f"{key}:{value}\n")
        file.write("\n")

        # [Events]
        file.write("[Events]\n")
        for line in d["[Events]"]:
            file.write(",".join([str(thing) for thing in line]) + "\n")
        file.write("\n")

        # [TimingPoints]
        file.write("[TimingPoints]\n")
        for line in d["[TimingPoints]"]:
            file.write(",".join([str(thing) for thing in line]) + "\n")
        file.write("\n\n")

        # [Colours]
        file.write("[Colours]\n")
        for key, value in d["[Colours]"].items():
            file.write(f"{key} : {value}\n")
        file.write("\n")

        # [HitObjects]
        file.write("[HitObjects]\n")
        for i, line in enumerate(d["[HitObjects]"]):
            # This extra if statement makes it so that the last line is NOT a newline.
            # I'm not sure if this would break things, but .osu files typically don't 
            # end in a newline, so better safe than sorry.
            if i != len(d["[HitObjects]"]) - 1:
                file.write(",".join([str(thing) for thing in line]) + "\n")
            else:
                file.write(",".join([str(thing) for thing in line]))

    return


def speedup_audio_file(input_path, rate, output_path):
    """Helper function for speeding up audio.
    Currently also pitches up audio; I want to avoid this.
    (Or maybe make it toggleable)
    """
    # input_audio = AudioSegment.from_file(input_path)
    #
    # sound_with_tempo = input_audio._spawn(input_audio.raw_data, overrides={
    #     "frame_rate": int(input_audio.frame_rate * rate)
    # })
    #
    # output_audio = sound_with_tempo.set_frame_rate(input_audio.frame_rate)
    #
    # output_audio.export(output_path, format="mp3")

    ff = ffmpy.FFmpeg(inputs={input_path: None}, outputs={output_path: ["-filter:a", f"atempo={rate}"]})
    ff.run()

    return


if __name__ == "__main__":
    # This is terrible!!!
    # I should make all of these options
    # Better yet, add a GUI!
    rate = 1.3
    d = convert_file_to_dict("./Sakamoto Maaya - Okaerinasai (tomatomerde Remix) (Azer) [Collab].osu")
    new_d = speedup_osu_file(d, rate)
    convert_dict_to_file(new_d, f"./Sakamoto Maaya - Okaerinasai (tomatomerde Remix) (Azer) [Collab {rate}x].osu")
    # print(convert_file_to_dict("./test.osu"))


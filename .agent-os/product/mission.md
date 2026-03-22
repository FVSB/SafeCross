# Mission

## What is SafeCross?

SafeCross is a computer vision system that helps visually impaired people cross
streets safely by detecting pedestrian traffic lights and crosswalks in real time.

## Problem

People with visual impairments face significant danger when crossing streets
independently. Existing aids (white canes, guide dogs) do not convey traffic
light state or crosswalk availability. SafeCross fills this gap using
accessible smartphone hardware.

## Solution

A three-model object detection pipeline (YOLOv8) that analyzes a single image
from the user's device and returns a binary decision:

- **Cross** — crosswalk visible + green light + no blocking vehicles
- **Wait / Ask for help** — any other condition

## Core Design Principle

**Zero False Positives.** The system must never tell a user to cross when it is
not safe. It is acceptable to be conservative (false negatives — saying "wait"
when crossing would actually be fine), but a false positive could cause injury
or death.

## Target User

Visually impaired pedestrians in urban environments. The system should work on
standard consumer smartphones without specialized hardware.

## Success Metrics

- 0 False Positives in field testing
- >90% precision on trained traffic light and crosswalk classes
- Inference time suitable for real-time use (<500ms on consumer GPU/CPU)
- Works in both daylight and nighttime conditions

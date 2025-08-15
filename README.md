


## Dataset

### EGOCOM

- **Overview**: EGOCOM contains egocentric conversation videos split into smaller clip parts, with aligned ground-truth annotations and a dataset-level transcript. Each video part has a matching joined-ground-truth JSON.

- **Directory layout**:
```text
EGOCOM/
├─ parts/
│  ├─ vid_001__day_1__con_1__person_1_part1(0_1920_social_interaction).MP4
│  ├─ vid_001__day_1__con_1__person_1_part1(1980_2370_social_interaction).MP4
│  └─ ...
├─ joined_ground_truth/
│  ├─ vid_001__day_1__con_1__person_1_part1(0_1920_social_interaction).json
│  ├─ vid_001__day_1__con_1__person_1_part1(1980_2370_social_interaction).json
│  └─ ...
└─ transcript/
   └─ ground_truth_transcript_with_frames.csv
```

- **Naming convention**:
  - **Video parts** in `EGOCOM/parts`: `vid_{vidId}__day_{dayId}__con_{conversationId}__person_{personId}_part{partIndex}({start}_{end}_{label}).MP4`
  - **Joined ground truth** in `EGOCOM/joined_ground_truth`: `{same_stem_as_part}.json`
  - The tuple inside parentheses encodes `{start}_{end}_{label}`. Units for `{start}` and `{end}` follow the dataset’s preprocessing (commonly frame indices); they are consistent between the MP4 filename and its JSON.

- **Pairing logic**:
  - For a part file with stem `S` (filename without extension), its ground truth JSON is `EGOCOM/joined_ground_truth/{S}.json`.
  - The dataset-level transcript is shared at `EGOCOM/transcript/ground_truth_transcript.csv`.

- **Joined ground truth JSON – structure**:
  - Top-level keys:
    - `metadata`: video-level info with either `video_name` (original) or `group_id` (prediction format), `analysis_type` (optional), `roi`, `frame_size_full`
    - `analysis_summary`: processing stats (`total_frames_processed`, `detected_frames`, `prediction_accuracy` for prediction analysis)
    - `frames`: array of per-frame annotations
  - Per-frame keys (each element of `frames`):
    - `frame_index`, `timestamp`, `social_category`
    - `red_circle`: `detected`, `position_full` [x,y], `position_content` [x,y], `radius`
    - `head_movement` and `next_movement`: each has `horizontal` and `vertical` with `radians` and `degrees` (may be null for first frame)
    - `speaker_id`
    - `face_detection`: array of boxes with `x1`,`y1`,`x2`,`y2`,`speaker_id` (optional)
    - `body_detection`: array of boxes with `x1`,`y1`,`x2`,`y2`,`speaker_id` (optional)
```json
{
  "metadata": {
    "group_id": "vid_001__day_1__con_1__person_1_part1(0_1920_social_interaction)",
    "analysis_type": "past_frame_prediction",
    "roi": { "x": 120, "y": 120, "w": 1280, "h": 720 },
    "frame_size_full": { "width": 1520, "height": 960 }
  },
  "analysis_summary": { 
    "total_frames_processed": 1914, 
    "detected_frames": 1914,
    "prediction_accuracy": {}
  },
  "frames": [
    {
      "frame_index": 0,
      "timestamp": 0.0,
      "social_category": "social_interaction",
      "red_circle": {
        "detected": true,
        "position_full": [759.0, 479.0],
        "position_content": [639.0, 359.0],
        "radius": 6
      },
      "head_movement": {
        "horizontal": { "radians": -0.00567232006898157, "degrees": -0.325 },
        "vertical": { "radians": -0.00567232006898157, "degrees": -0.325 }
      },
      "next_movement": {
        "horizontal": { "radians": -0.00567232006898157, "degrees": -0.325 },
        "vertical": { "radians": 0.014180800172453928, "degrees": 0.8125 }
      }
    }
  ]
}
```

- **Transcript CSV – minimal expectations**:
  - Located at `EGOCOM/transcript/ground_truth_transcript.csv`.
  - **Schema (columns)**:
    - `conversation_id` (str): conversation/part identifier, e.g., `day_1__con_1__part1`
    - `startTime` (float, seconds): word start time
    - `endTime` (float, seconds): word end time
    - `speaker_id` (int): speaker identifier for the word
    - `word` (str): token text (may include punctuation, e.g., `have.`)
    - `frame` (list[int] or stringified list): frame indices aligned to the word, e.g., `[5,6,7]`
  - **Example row (CSV)**:
```csv
conversation_id,endTime,speaker_id,startTime,word,frame
day_1__con_1__part1,1.14,1,1.011,have.,"[5,6,7]"
```
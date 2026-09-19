# Credits — the practise photographs

Every photograph here is a **NASA** image. Works of the United States federal government carry no
copyright (17 U.S.C. §105), so these are **public domain**: no permission and no licence is needed
for any use. NASA's media guidelines ask to be credited, which is what the table below is for. Each
is at most 256 pixels on its long side.

They are all one crew — **STS-135**, the last Space Shuttle mission: Christopher Ferguson, Douglas
Hurley, Sandra Magnus and Rex Walheim — photographed together through training, press and flight,
with the Expedition 28 crew who watched the launch. That is the point of the set: the same few faces
over and over, so naming them teaches the model a face rather than a collection of strangers, and it
can start getting them right. **Both men and women are in it.**

## How these eight were chosen

Not by eye, and not by counting boxes. A **strict, independent** face detector — dlib's, through
`face_recognition` — was the referee: it was asked where the faces are, and every box the page's own
finder draws was then compared with it.

1. Of **1,232** NASA photographs of the mission, **100** hold two or three faces big enough to work
   with (18 pixels or more, near-frontal).
2. Each was cropped four ways, so the faces fill between 68 and 100 per cent of the frame →
   **385 crops**.
3. Every crop was pushed through the page's own finder, at the size the page really searches at
   (`ANALYSIS` in `src/image.ts`), exactly as an uploaded picture is.
4. Kept: crops where it draws **two or three boxes and every box lands on a face**. **37 photographs
   passed**; these eight are the pick, for variety of moment and a mix of men and women.

**Cropping to the people is what makes the count stable.** A face has to be a decent share of the
frame, and a crew photograph straight out of the camera is mostly room. At 58 per cent face coverage,
most of these pictures came back with one box or none; at about 78 per cent they come back with two
or three.

| File | Photograph | Date | NASA | Boxes | Source |
|---|---|---|---|---|---|
| `01-suiting-up.jpg` | Crew and training team during bailout training | 2010-12-03 | jsc2010e195531 | 3 | [images.nasa.gov](https://images.nasa.gov/details-jsc2010e195531) |
| `02-in-the-training-room.jpg` | Expedition 28 crew member and JAXA astronaut Satoshi Furukawa | 2011-01-25 | jsc2011e006403 | 3 | [images.nasa.gov](https://images.nasa.gov/details-jsc2011e006403) |
| `03-press-conference.jpg` | The crew at a press conference | 2011-07-07 | KSC-2011-5102 | 3 | [images.nasa.gov](https://images.nasa.gov/details-KSC-2011-5102) |
| `04-press-conference-two.jpg` | STS-135 press conference | 2011-06-30 | jsc2011e060451 | 2 | [images.nasa.gov](https://images.nasa.gov/details-jsc2011e060451) |
| `05-the-briefing.jpg` | STS-135 press conference, from the other side | 2011-06-30 | jsc2011e060424 | 2 | [images.nasa.gov](https://images.nasa.gov/details-jsc2011e060424) |
| `06-watching-the-launch.jpg` | Expedition 28 crew members watch the launch of STS-135 | 2011-07-08 | iss028e014701 | 2 | [images.nasa.gov](https://images.nasa.gov/details-iss028e014701) |
| `07-water-survival-training.jpg` | STS-135 water survival training at the Neutral Buoyancy Laboratory | 2011-02-09 | jsc2011e016240 | 2 | [images.nasa.gov](https://images.nasa.gov/details-jsc2011e016240) |
| `08-russia-training.jpg` | STS-135 training in Russia | 2011-03-29 | jsc2011e040319 | 2 | [images.nasa.gov](https://images.nasa.gov/details-jsc2011e040319) |

Eight photographs, 63 KB in total. The list the page reads is `PRACTISE_PHOTOS` in `src/samples.ts`;
**the two must stay in step** — a file named here and not in the list is dead weight, and a file in
the list and not here is an unexplained face.

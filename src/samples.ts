/**
 * samples.ts — the photographs the page can hand you.
 *
 * The demo asks you to bring your own pictures, and it should. But an empty page with a
 * file dialog is a poor first minute: nothing at all happens until you have gathered
 * twenty photographs. So the page also carries a small set of its own, and hands over all
 * of them, shuffled.
 *
 * They are **real photographs, not drawings**, and they are **one crew**: NASA's STS-135 —
 * Christopher Ferguson, Douglas Hurley, Sandra Magnus and Rex Walheim — photographed
 * together through training, press and flight.
 *
 * **The same few faces over and over is the point of the set.** George, 2026-09-19: *"it
 * need to able to guess correctly, so the boxes should have similarities"*. Eight
 * strangers teach a network nothing that transfers; one crew teaches it four faces it meets
 * again and again, which is a thing it can actually start getting right. Both men and women
 * are in it, and every picture holds two or three boxes.
 *
 * NASA's images are public domain — works of the United States federal government carry
 * no copyright (17 U.S.C. §105) — so nothing here needs a licence or an attribution, and
 * the photograph, its NASA id and its date are recorded per picture below and in
 * `site/photos/CREDITS.md` anyway.
 *
 * Every picture was measured with the page's own face finder before it was chosen, and
 * each sits at or under 256 pixels on its long side, which keeps the whole set to about
 * a hundred and seventy kilobytes.
 */

/** One photograph, and where it came from. */
export interface PractisePhoto {
  /** The file in `site/photos/`. */
  file: string;
  /** What the picture is. */
  title: string;
  /** Who to credit, and which NASA frame it is. */
  credit: string;
  /** The NASA page it came from. */
  page: string;
}

/** Where the photographs live, relative to the page. */
const FOLDER = './photos/';

/**
 * The set: one crew, eight pictures.
 *
 * Chosen out of 1,232 NASA photographs of the same mission, and chosen on **one question only:
 * where the page's own face finder draws its boxes.** A strict, independent detector — dlib's, via
 * `face_recognition` — was the referee: 100 photographs held two or three faces big enough to
 * matter, each was cropped four ways, and the 385 crops were pushed through the page's own finder
 * exactly as a real picture is. The eight here are crops where it draws **two or three boxes and
 * every box lands on a face** (measured overlap with the referee's face, 0.38 to 0.89).
 *
 * That measurement is also what fixed the finder: 37 of the 385 crops passed, and of the boxes it
 * drew across all of them **95% were on a face** — but it only got there once the search stepped
 * one pixel at a time (`haar.ts`). Before that it found roughly a third of the faces that are
 * really in these pictures. **Do not trust a count alone here** — a box on a torso still counts as
 * a box, which is how an earlier version of this set shipped with boxes on people's bodies.
 *
 * Every one is **cropped to the people in it**, and that is what makes the count stable: the finder
 * needs a face to be a decent share of the frame, and a crew photograph straight out of the camera
 * is mostly room. The crops here fill about four fifths of the frame with faces — at half that, most
 * of these pictures came back with one box or none.
 */
export const PRACTISE_PHOTOS: PractisePhoto[] = [
  { file: '01-suiting-up.jpg', title: 'Crew and training team during bailout training', credit: 'NASA · jsc2010e195531 · 3 December 2010', page: 'https://images.nasa.gov/details-jsc2010e195531' },
  { file: '02-in-the-training-room.jpg', title: 'Expedition 28 crew member and JAXA astronaut Satoshi Furukawa', credit: 'NASA · jsc2011e006403 · 25 January 2011', page: 'https://images.nasa.gov/details-jsc2011e006403' },
  { file: '03-press-conference.jpg', title: 'The crew at a press conference', credit: 'NASA · KSC-2011-5102 · 7 July 2011', page: 'https://images.nasa.gov/details-KSC-2011-5102' },
  { file: '04-press-conference-two.jpg', title: 'STS-135 press conference', credit: 'NASA · jsc2011e060451 · 30 June 2011', page: 'https://images.nasa.gov/details-jsc2011e060451' },
  { file: '05-the-briefing.jpg', title: 'STS-135 press conference, from the other side', credit: 'NASA · jsc2011e060424 · 30 June 2011', page: 'https://images.nasa.gov/details-jsc2011e060424' },
  { file: '06-watching-the-launch.jpg', title: 'Expedition 28 crew members watch the launch of STS-135', credit: 'NASA · iss028e014701 · 8 July 2011', page: 'https://images.nasa.gov/details-iss028e014701' },
  { file: '07-water-survival-training.jpg', title: 'STS-135 water survival training at the Neutral Buoyancy Laboratory', credit: 'NASA · jsc2011e016240 · 9 February 2011', page: 'https://images.nasa.gov/details-jsc2011e016240' },
  { file: '08-russia-training.jpg', title: 'STS-135 training in Russia', credit: 'NASA · jsc2011e040319 · 29 March 2011', page: 'https://images.nasa.gov/details-jsc2011e040319' },
];

/**
 * A random handful, as files.
 *
 * Random rather than the same ten every time, so a second run is not a memory test. They
 * leave here as `File`s and are handed to exactly the code an upload goes through — the
 * queue, the face search, one box named at a time — because a practise set that skipped
 * any of that would not be practising the page.
 *
 * `random` is a parameter so that a test can hand in a seeded generator and get the same
 * ten twice.
 */
export async function makeSampleFiles(count = 10, random: () => number = Math.random): Promise<File[]> {
  const pool = [...PRACTISE_PHOTOS];
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    const swap = pool[i];
    pool[i] = pool[j];
    pool[j] = swap;
  }

  const out: File[] = [];
  for (const photo of pool.slice(0, Math.max(0, count))) {
    try {
      const response = await fetch(`${FOLDER}${photo.file}`);
      if (!response.ok) continue;
      const blob = await response.blob();
      out.push(new File([blob], photo.file, { type: blob.type || 'image/jpeg' }));
    } catch {
      // One photograph that will not load is not worth losing the other nine over.
    }
  }
  return out;
}


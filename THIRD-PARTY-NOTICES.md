# Third-party notices

This project redistributes one thing it did not create. **The notice is kept as its licence
requires**, and it is kept twice on purpose: inside the generated file a reader may never open, and
here, where a reader looks.

## Rainer Lienhart's frontal-face cascade, as shipped with OpenCV

| | |
|---|---|
| what it is | a stump-based 24×24 AdaBoost frontal-face detector |
| where it came from | `haarcascade_frontalface_default.xml`, from OpenCV's `data/haarcascades` |
| what is copied | the **trained numbers only** — 25 stages, 2913 features, 6383 rectangles, packed as typed arrays |
| what is not copied | the detector. `src/haar.ts` is written here from the published Viola-Jones algorithm, and its arithmetic was checked against OpenCV 4.10 before it was used |
| how it is regenerated | `npm run cascade` — `node tools/build-cascade.mjs [path-to-xml]`, which writes both `src/cascade-data.ts` and this file |
| why the numbers are packed | the XML is ~930 KB of pretty-printed decimal; emitted as typed arrays it is about 160 KB and every value goes back through the same `Float32` OpenCV stores it in |

### The licence, exactly as the cascade file carries it

```
Stump-based 24x24 discrete(?) adaboost frontal face detector.
    Created by Rainer Lienhart.

////////////////////////////////////////////////////////////////////////////////////////

  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

  By downloading, copying, installing or using the software you agree to this license.
  If you do not agree to this license, do not download, install,
  copy or use the software.


                        Intel License Agreement
                For Open Source Computer Vision Library

 Copyright (C) 2000, Intel Corporation, all rights reserved.
 Third party copyrights are property of their respective owners.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

   * Redistribution's of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

   * Redistribution's in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.

   * The name of Intel Corporation may not be used to endorse or promote products
     derived from this software without specific prior written permission.

 This software is provided by the copyright holders and contributors "as is" and
 any express or implied warranties, including, but not limited to, the implied
 warranties of merchantability and fitness for a particular purpose are disclaimed.
 In no event shall the Intel Corporation or contributors be liable for any direct,
 indirect, incidental, special, exemplary, or consequential damages
 (including, but not limited to, procurement of substitute goods or services;
 loss of use, data, or profits; or business interruption) however caused
 and on any theory of liability, whether in contract, strict liability,
 or tort (including negligence or otherwise) arising in any way out of
 the use of this software, even if advised of the possibility of such damage.
```

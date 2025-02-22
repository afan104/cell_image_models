// // description: photoshop script for parsing json files to image
#include utils.js // commented out for formatting
#include json2.js

closeAll()
const currDir = "~/Desktop/capaldi/"
const scale = 1608 / 2235

// sanity check the masks are matching
sanityCheckMasks(currDir, scale)

// overlay img and masks in order of name
drawMasksInOrder(currDir)

// complete and close options


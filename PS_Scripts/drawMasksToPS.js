// // description: photoshop script for parsing json files to image
#include utils.js // commented out for formatting
#include json2.js

closeAll()
const dataDir = "~/Desktop/capaldi/"
const pixelToPointConversion = 1608 / 2235

// sanity check the masks are matching
sanityCheckMasks(dataDir, pixelToPointConversion)

// overlay img and masks in order of name
drawMasksInOrder(dataDir, pixelToPointConversion)
// // description: photoshop script for parsing json files to image
#include utils.js // commented out for formatting
#include json2.js

closeAll()
const currDir = "~/Desktop/capaldi/"
const scale = 1608 / 2235
const validMaskTypes = ["normal", "kog1", "both"]
var validMask = false

// sanity check the masks are matching
var sanityCheck = confirm("check if masks match?")
while (sanityCheck) {
  var maskType = prompt("Choose mask option: normal, kog1, both")

  // check if valid mask
  for (var i = 0; i < validMaskTypes.length; i++) {
    if (maskType == validMaskTypes[i]) {
      validMask = true
      break
    }
  }

  // implement mask if valid
  if (validMask) {
    compareMasksImgs(currDir, maskType, scale)
  } else {
    alert("please type a valid option")
  }
  sanityCheck = confirm("check if masks match, again?")
}

// complete and close options
alert("sanity check completed")
var close = confirm("close all images?")
if (close) {
  closeAll()
}

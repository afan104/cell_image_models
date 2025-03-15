#include json2.js
#include utils.js

const repoDir = File($.fileName).parent.parent
const dataDir = new Folder(repoDir + "/psd_imgs/")
const dataFiles = dataDir.getFiles("*.psd")

// get list of existing json annotations
var annotationTrackerData = readJsonFile(
  repoDir + "/Data/annotation_tracker.json"
)
const existingFiles = annotationTrackerData["3-21"]

// tracker date for newly created json to be recorded under
const trackerDate = "3-22"
annotationTrackerData[trackerDate] = existingFiles

// go through all psd files
for (var i = 0; i < dataFiles.length; i++) {
  // if psd already has a json-stored annotation, skip
  var seen = false
  var fileName = dataFiles[i].name.slice(0, -4) // remove .psd extension
  alert("attempting to annotate " + fileName)
  for (var j = 0; j < existingFiles.length; j++) {
    if (fileName == existingFiles[j]) {
      seen = true
      alert("this file previously annotated:" + fileName)
      break
    }
  }
  if (seen) {
    continue
  } else {
    // open document
    app.open(dataFiles[i])
    // access doc
    var doc = app.activeDocument
    var outputDir = new Folder(repoDir + "/newdata").absoluteURI
    const outputPath = new File(outputDir + "/" + fileName + ".json")
    img_to_json(doc, outputPath, annotationTrackerData, trackerDate)
  }
}
alert("updating tracker file into new 'test' file")
annotationTrackerFile = new File(repoDir + "/Data/annotation_tracker_test.json")
annotationTrackerFile.open("w")
annotationTrackerFile.write(JSON.stringify(annotationTrackerData))
annotationTrackerFile.close()
alert("done updating tracker file")
const close = confirm("close all open documents?")
if (close) {
  closeAll()
}

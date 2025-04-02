#include json2.js
#include utils.js

const repoDir = File($.fileName).parent.parent
const dataDir = new Folder(repoDir + "/correction/")
const dataFiles = dataDir.getFiles("*.psd")

// get list of existing json annotations
var annotationTrackerData = readJsonFile(
  repoDir + "/Data/annotation_tracker.json"
)
const existingFiles = annotationTrackerData["3-21"]
alert("length of existing files: " + annotationTrackerData["3-21"].length)

// tracker date for newly created json to be recorded under
const trackerDate = "4-2"
annotationTrackerData[trackerDate] = JSON.parse(JSON.stringify(existingFiles))
// go through all psd files
for (var i = 0; i < dataFiles.length; i++) {
  // if psd already has a json-stored annotation, skip
  var seen = false
  var fileName = dataFiles[i].name.slice(0, -4) // remove .psd extension
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
    var outputDir = new Folder(repoDir + "/correction").absoluteURI
    var outputPath = new File(outputDir + "/" + fileName + ".json")
    img_to_json(doc, outputPath, annotationTrackerData, trackerDate)
  }
}
alert("length of prev files: "+ annotationTrackerData["3-21"].length)
alert("length of current files: " + annotationTrackerData["4-1"].length)
annotationTrackerFile = new File(repoDir + "/Data/annotation_tracker.json")
annotationTrackerFile.open("w")
annotationTrackerFile.write(JSON.stringify(annotationTrackerData, null, 2))
annotationTrackerFile.close()
alert("done updating tracker file")
const close = confirm("close all open documents?")
if (close) {
  closeAll()
}

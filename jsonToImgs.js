// description: photoshop script for parsing json files to image
#include utils.js // commented out for formatting
#include json2.js

// initialize doc reference
var doc = app.activeDocument
// read in json and save info into dictionary
var dataPath = "~/Desktop/capaldi/jsonFiles/"
var jsonFilePath = dataPath + doc.name.slice(0, -4) + ".json"
var jsonShapeData = readJsonFile(jsonFilePath)["shapes"]

// grab all the paths from json
var pathItemsDict = {"normal": [], "kogg": []}  // value is a list of shapes; shape is a list of points; 3D tensor-like shape
for (var i = 0; i < jsonShapeData.length; i++) {
  var shapeData = jsonShapeData[i]
  var layer = shapeData["label"]
  var shape = shapeData["points"]

  pathItemsDict[layer].push(shape)
}

// TODO: initialize layers with vector masks
alert("1")
createVectorLayer("normal")
createVectorLayer("kogg")
alert("3")

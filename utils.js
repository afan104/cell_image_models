// read json file and return json object
function readJsonFile(jsonFilePath) {
  var file = File(jsonFilePath)

  // check if file exists
  if (!file.exists) {
    alert("File does not exist")
    return
  }

  // open file and read content
  file.open("r")
  var content = file.read()
  file.close()

  // attempt to parse and return json content
  try {
    return JSON.parse(content)
  } catch (e) {
    alert("Error parsing json file")
    return
  }
}

// create new layer function
createVectorLayer = function (doc, shapesDict, name) {
  // create new layer
  var layerRef = doc.artLayers.add()
  layerRef.name = name

  // set as active layer
  doc.activeLayer = layerRef

  // create path item containing all active layer shapes and change it to a vector mask
  var subpathItems = shapesDict[name]
  var pathItem = doc.pathItems.add(subpathItems)
  pathItem.kind = PathKind.VECTORMASK

  alert("2")
  // link the path item to the layer
  layerRef.pathItems = pathItem
  alert("3")
}

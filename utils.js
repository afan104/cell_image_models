/////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////*SIMPLE UTILITY FUNCTIONS*////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

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

// close all files
function closeAll() {
  while (app.documents.length) {
    app.activeDocument.close(SaveOptions.DONOTSAVECHANGES)
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////*DRAWING SHAPES IN PHOTOSHOP*//////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

// function to create pathPointInfo object
function getPathPointInfo(pointPosition) {
  var ppi = new PathPointInfo()
  ppi.kind = PointKind.CORNERPOINT
  ppi.anchor = pointPosition
  ppi.leftDirection = pointPosition
  ppi.rightDirection = pointPosition
  return ppi
}

// function to create subpathInfo object
function getSubPathInfo(shape, scale) {
  var entireSubPath = []

  for (var i = 0; i < shape.length; i++) {
    var point = shape[i]
    point[0] = point[0] * scale
    point[1] = point[1] * scale

    var ppi = getPathPointInfo(point)

    // add to subpath
    entireSubPath.push(ppi)
  }

  // define SubPathInfo as an object
  var spi = new SubPathInfo()
  spi.entireSubPath = entireSubPath
  spi.closed = true
  spi.operation = ShapeOperation.SHAPEADD
  return spi
}

// function to create and add PathItem to vector mask layer linked to artlayer
function addPathItem(doc, name, shapesDict, scale) {
  // get all PathItem shapes (array of subpathinfo objects)
  var spiArray = []
  var shapes = shapesDict[name]

  for (var i = 0; i < shapes.length; i++) {
    var shape = shapes[i]
    var spi = getSubPathInfo(shape, scale)
    spiArray.push(spi)
  }
  var pathItem = doc.pathItems.add(name, spiArray)
  pathItem.PathKind = PathKind.VECTORMASK
  return pathItem
}

// function to visualize json shapes onto an image
function jsonToImgs(doc, jsonFilePath, maskType, scale) {
  var jsonShapeData = readJsonFile(jsonFilePath)["shapes"]

  // grab all the paths from json
  var shapesDict = { normal: [], kogg: [] } // value is a list of shapes; shape is a list of points; 3D tensor-like shape
  for (var i = 0; i < jsonShapeData.length; i++) {
    var shapeData = jsonShapeData[i]
    var layer = shapeData["label"]
    var shape = shapeData["points"]

    shapesDict[layer].push(shape)
  }

  // Add paths to each layer
  if (maskType == "kog1") {
    var kog1PathItem = addPathItem(doc, "kogg", shapesDict, scale)
    var out = { kog1PathItem: kog1PathItem }
  }
  if (maskType == "normal") {
    var normalPathItem = addPathItem(doc, "normal", shapesDict, scale)
    var out = { normalPathItem: normalPathItem }
  }
  if (maskType == "both") {
    var kog1PathItem = addPathItem(doc, "kogg", shapesDict, scale)
    var normalPathItem = addPathItem(doc, "normal", shapesDict, scale)
    var out = { kog1PathItem: kog1PathItem, normalPathItem: normalPathItem }
  }
  return out
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////*COMPLEX IMG & MASK UTILITY*///////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

// function that visualizes masks on images and allows you to visually record matches
function compareMasksImgs(currDir, maskType, scale) {
  // get files
  const imgDir = currDir + "imgFiles/"
  const jsonDir = currDir + "jsonFiles/"
  const imgFiles = Folder(imgDir).getFiles("*.png")
  const jsonFiles = Folder(jsonDir).getFiles("*.json")

  // record matches
  var record = String(maskType) + " results"

  // check by image
  var matchedJson = [] // tracks matched json files
  for (var i = 0; i < imgFiles.length; i++) {
    record = record + "\n\n" + String(imgFiles[i].name) + ":\n"

    // open doc
    app.open(imgFiles[i])
    var doc = app.activeDocument

    // compare to json masks
    for (var j = 0; j < jsonFiles.length; j++) {
      // skip if json mask already matched
      if (j in matchedJson) {
        continue
      }

      // draw target mask path
      var pathItems = jsonToImgs(doc, String(jsonFiles[j]), maskType, scale)

      // record matching pairs
      var isSame = confirm(
        String(j + 1) +
          "/" +
          String(jsonFiles.length) +
          ": Do the masks match the image?"
      )
      if (isSame) {
        record = record + String(jsonFiles[j].name)
        matchedJson.push(j)
        break
      } else {
        // clear masks if not a match
        if (maskType == "kog1") {
          pathItems.kog1PathItem.remove()
        }
        if (maskType == "normal") {
          pathItems.normalPathItem.remove()
        }
        if (maskType == "both") {
          pathItems.normalPathItem.remove()
          pathItems.kog1PathItem.remove()
        }
      }
    }
  }
  alert(record)
  return record
}

// .include and .has not supported in photoshop
function chooseMaskType() {
  const validMaskTypes = ["both", "normal", "kog1"]
  var maskType = prompt("Choose mask option: normal, kog1, both")
  var validMaskType = false

  while (!validMaskType) {
    // check if valid masktype
    for (var i = 0; i < validMaskTypes.length; i++) {
      if (maskType == validMaskTypes[i]) {
        validMaskType = true
        break
      }
    }

    // retry prompt if invalid
    if (!validMaskType) {
      maskType = prompt(
        "Invalid maskType\nChoose mask option: normal, kog1, both"
      )
    }
  }
  return maskType
}

// optional: sanity check prompt that compares all masks to imgs until correct pairs found
function sanityCheckMasks(currDir, scale) {
  var skipSanityCheck = confirm("skip sanity check?")
  while (!skipSanityCheck) {
    var maskType = chooseMaskType()
    compareMasksImgs(currDir, maskType, scale)
    alert("sanity check completed")
    skipsanityCheck = confirm("skip repeat sanity check?")
  }
  closeAll()
}

function drawMasksInOrder(currDir) {
  alert("drawing masks")
  // get files
  const imgDir = currDir + "imgFiles/"
  const jsonDir = currDir + "jsonFiles/"
  const imgFiles = Folder(imgDir).getFiles("*.png")
  const jsonFiles = Folder(jsonDir).getFiles("*.json")
  const maskType = chooseMaskType()

  // go through each image
  for (var i = 0; i < imgFiles.length; i++) {
    // open doc
    app.open(imgFiles[i])
    var doc = app.activeDocument

    // draw target mask path
    jsonToImgs(doc, String(jsonFiles[i]), maskType, scale)
  }
  alert("all masks completed")
  var open = confirm("keep all images open?")
  if (!open) {
    closeAll()
  }
}

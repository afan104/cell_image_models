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

// function to create pathPointInfo object
function getPathPointInfo(pointPosition) {
  var ppi = new PathPointInfo()
  ppi.kind = PointKind.CORNERPOINT
  ppi.anchor = pointPosition
  ppi.leftDirection = pointPosition
  ppi.RightDirection = pointPosition
  return ppi
}

// function to create subpathInfo object
function getSubPathInfo(shape) {
  var entireSubPath = []

  for (i = 0; i < shape.length; i++) {
    var point = shape[i]
    var ppi = getPathPointInfo(point)

    // add to subpath
    entireSubPath.push(ppi)
    alert("adding point to subpath")
  }

  // define SubPathInfo as an object
  var spi = new SubPathInfo()
  spi.entireSubPath = entireSubPath
  spi.closed = true
  return spi
}

// function to create and add PathItem
function addPathItem(doc, name, shapesDict) {
  // create layer for vector mask and set active layer
  var layer = doc.artLayers.add()
  layer.name = name
  doc.activeLayer = layer

  // get all shapes (array of subpathinfo objects)
  var spiArray = []
  var shapes = shapesDict[name]
  shapes = [
    [
      [1, 2],
      [3, 5],
      [, 4, 6],
    ],
    [
      [7, 8],
      [9, 10],
      [11, 12],
    ],
  ] //debug

  for (var i = 0; i < shapes.length; i++) {
    var shape = shapes[i]
    var spi = getSubPathInfo(shape)
    spiArray.push(spi)
    alert("added subpath info to spiArray")
    if (spiArray.length == 2) {
      break // get two shapes
    }
  }

  alert("checking spiArray contains subpath info")
  for (var i = 0; i < spiArray.length; i++) {
    for (var j = 0; j < spiArray[i].entireSubPath.length; j++) {
      alert("anchor")
      alert(spiArray[i].entireSubPath[j].anchor)
      alert(typeof spiArray[i].entireSubPath[j].anchor[1])
      alert("left")
      alert(spiArray[i].entireSubPath[j].leftDirection)
      alert("right")
      alert(spiArray[i].entireSubPath[j].rightDirection)
      alert("kind")
      alert(spiArray[i].entireSubPath[j].kind)
    }
  }
  // Check the contents of spiArray
  var pathItem = doc.pathItems.add(name, spiArray)
  alert(2)
  pathItem.PathKind = PathKind.VECTORMASK

  // add stroke to visualize
  pathItem.strokePath(ToolType.PENCIL)
}

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
  ppi.rightDirection = pointPosition
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
  }

  // define SubPathInfo as an object
  var spi = new SubPathInfo()
  spi.entireSubPath = entireSubPath
  spi.closed = true
  spi.operation = ShapeOperation.SHAPEADD
  return spi
}

// function to create and add PathItem to vector mask layer linked to artlayer
function addPathItem(doc, name, shapesDict) {
  // get all PathItem shapes (array of subpathinfo objects)
  var spiArray = []
  var shapes = shapesDict[name]

  for (var i = 0; i < shapes.length; i++) {
    var shape = shapes[i]
    var spi = getSubPathInfo(shape)
    spiArray.push(spi)
  }

  var pathItem = doc.pathItems.add(name, spiArray)
  pathItem.PathKind = PathKind.VECTORMASK
}

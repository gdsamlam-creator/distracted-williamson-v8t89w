import { useState, useEffect, useRef } from 'react';

// Helper function to calculate distance between two points
const getDistance = (p1: { x: number; y: number }, p2: { x: number; y: number }): number => {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  return Math.sqrt(dx * dx + dy * dy);
};

// Helper function to calculate polygon area using the shoelace formula
const getPolygonArea = (points: { x: number; y: number }[]): number => {
  let area = 0;
  const n = points.length;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    area += points[i].x * points[j].y;
    area -= points[j].x * points[i].y;
  }
  return Math.abs(area) / 2;
};

// Simplified edge detection that won't cause infinite loops
const detectEdges = (imageData: ImageData): { points: { x: number; y: number }[]; edges: boolean[][] } => {
  const width = imageData.width;
  const height = imageData.height;
  const data = imageData.data;
  
  // Convert to grayscale
  const grayscale = new Float32Array(width * height);
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    grayscale[i / 4] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  
  // Sobel kernels
  const kernelX = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
  ];
  
  const kernelY = [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
  ];
  
  // Apply Sobel operator
  const gradient = new Float32Array(width * height);
  const edges = Array(height).fill(0).map(() => Array(width).fill(false));
  
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let gx = 0;
      let gy = 0;
      
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const pixel = grayscale[(y + ky) * width + (x + kx)];
          gx += pixel * kernelX[ky + 1][kx + 1];
          gy += pixel * kernelY[ky + 1][kx + 1];
        }
      }
      
      const magnitude = Math.sqrt(gx * gx + gy * gy);
      gradient[y * width + x] = magnitude;
      
      // Apply threshold
      if (magnitude > 40) {
        edges[y][x] = true;
      }
    }
  }
  
  // Find contour points using a grid sampling approach instead of contour tracing
  const points: { x: number; y: number }[] = [];
  const gridSpacing = 20; // Sample every 20 pixels
  
  // First, find bounding box of edges
  let minX = width, minY = height, maxX = 0, maxY = 0;
  let edgeCount = 0;
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (edges[y][x]) {
        edgeCount++;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  
  // If no edges detected, return default polygon
  if (edgeCount === 0 || (maxX - minX) < 10 || (maxY - minY) < 10) {
    console.log("No edges found, using default polygon");
    return {
      points: [
        { x: 0.2, y: 0.3 },
        { x: 0.8, y: 0.2 },
        { x: 0.7, y: 0.7 },
        { x: 0.3, y: 0.8 },
      ],
      edges
    };
  }
  
  // Add padding to bounding box
  const padding = Math.min(width, height) * 0.05;
  minX = Math.max(0, minX - padding);
  minY = Math.max(0, minY - padding);
  maxX = Math.min(width - 1, maxX + padding);
  maxY = Math.min(height - 1, maxY + padding);
  
  // Sample points along the grid
  for (let y = minY; y <= maxY; y += gridSpacing) {
    for (let x = minX; x <= maxX; x += gridSpacing) {
      // Check if this point is near an edge
      let edgeNearby = false;
      const searchRadius = 5;
      
      for (let dy = -searchRadius; dy <= searchRadius && !edgeNearby; dy++) {
        for (let dx = -searchRadius; dx <= searchRadius && !edgeNearby; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx >= 0 && nx < width && ny >= 0 && ny < height && edges[ny][nx]) {
            edgeNearby = true;
          }
        }
      }
      
      if (edgeNearby) {
        // Find the nearest edge point
        let nearestDist = Infinity;
        let nearestX = x;
        let nearestY = y;
        
        for (let dy = -10; dy <= 10; dy++) {
          for (let dx = -10; dx <= 10; dx++) {
            const nx = x + dx;
            const ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height && edges[ny][nx]) {
              const dist = Math.abs(dx) + Math.abs(dy);
              if (dist < nearestDist) {
                nearestDist = dist;
                nearestX = nx;
                nearestY = ny;
              }
            }
          }
        }
        
        points.push({ x: nearestX / width, y: nearestY / height });
      }
    }
  }
  
  // If we have enough points, sort them in clockwise order
  if (points.length >= 3) {
    // Find center
    const centerX = points.reduce((sum, p) => sum + p.x, 0) / points.length;
    const centerY = points.reduce((sum, p) => sum + p.y, 0) / points.length;
    
    // Sort by angle
    points.sort((a, b) => {
      const angleA = Math.atan2(a.y - centerY, a.x - centerX);
      const angleB = Math.atan2(b.y - centerY, b.x - centerX);
      return angleA - angleB;
    });
    
    // Simplify polygon to reduce points
    const simplified = simplifyPolygon(points, 0.01);
    console.log(`Detected ${points.length} edge points, simplified to ${simplified.length} points`);
    return { points: simplified, edges };
  }
  
  // Fallback: use bounding box points
  console.log("Using bounding box as fallback");
  return {
    points: [
      { x: minX / width, y: minY / height },
      { x: maxX / width, y: minY / height },
      { x: maxX / width, y: maxY / height },
      { x: minX / width, y: maxY / height },
    ],
    edges
  };
};

// Simplify polygon using Douglas-Peucker algorithm
const simplifyPolygon = (points: { x: number; y: number }[], epsilon: number): { x: number; y: number }[] => {
  if (points.length <= 2) return points;
  
  // Find the point with maximum distance
  let dmax = 0;
  let index = 0;
  const end = points.length - 1;
  
  for (let i = 1; i < end; i++) {
    const d = perpendicularDistance(points[i], points[0], points[end]);
    if (d > dmax) {
      index = i;
      dmax = d;
    }
  }
  
  // If max distance is greater than epsilon, recursively simplify
  if (dmax > epsilon) {
    const recResults1 = simplifyPolygon(points.slice(0, index + 1), epsilon);
    const recResults2 = simplifyPolygon(points.slice(index), epsilon);
    
    // Build the result list
    return [...recResults1.slice(0, -1), ...recResults2];
  } else {
    return [points[0], points[end]];
  }
};

const perpendicularDistance = (point: { x: number; y: number }, lineStart: { x: number; y: number }, lineEnd: { x: number; y: number }): number => {
  const area = Math.abs(
    (lineEnd.x - lineStart.x) * (lineStart.y - point.y) - 
    (lineStart.x - point.x) * (lineEnd.y - lineStart.y)
  );
  const lineLen = Math.sqrt(
    (lineEnd.x - lineStart.x) ** 2 + (lineEnd.y - lineStart.y) ** 2
  );
  return area / lineLen;
};

export default function PaperPatternScanner() {
  const [capturedImageSrc, setCapturedImageSrc] = useState<string | null>(null);
  const [processedImageSrc, setProcessedImageSrc] = useState<string | null>(null);
  const [measurements, setMeasurements] = useState<{
    area: number | null;
    perimeter: number | null;
    sideLengths: number[] | null;
  }>({ area: null, perimeter: null, sideLengths: null });
  const [cameraDistance, setCameraDistance] = useState<number>(30);
  const [showCameraStream, setShowCameraStream] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [availableCameras, setAvailableCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [editMode, setEditMode] = useState<boolean>(false);
  const [polygonPoints, setPolygonPoints] = useState<{x: number; y: number}[]>([
    { x: 0.2, y: 0.3 },
    { x: 0.8, y: 0.2 },
    { x: 0.7, y: 0.7 },
    { x: 0.3, y: 0.8 },
  ]);
  const [activePointIndex, setActivePointIndex] = useState<number | null>(null);
  const [imageDimensions, setImageDimensions] = useState<{width: number; height: number}>({width: 0, height: 0});
  const [cameraScale, setCameraScale] = useState<{scaleX: number; scaleY: number}>({scaleX: 1, scaleY: 1});
  const [edgeDetectionMode, setEdgeDetectionMode] = useState<boolean>(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const editCanvasRef = useRef<HTMLCanvasElement>(null);
  const edgeCanvasRef = useRef<HTMLCanvasElement>(null);
  const [isDetectingEdges, setIsDetectingEdges] = useState<boolean>(false);

  // Load available cameras
  useEffect(() => {
    const loadCameras = async () => {
      try {
        await navigator.mediaDevices.getUserMedia({ video: true });
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        setAvailableCameras(videoDevices);
        if (videoDevices.length > 0) {
          setSelectedCamera(videoDevices[0].deviceId);
        }
      } catch (err) {
        console.log("Camera permission not granted yet");
      }
    };
    loadCameras();
  }, []);

  // Start/stop camera stream
  useEffect(() => {
    if (showCameraStream && selectedCamera) {
      startCameraStream();
    } else {
      stopCameraStream();
    }
    
    return () => {
      stopCameraStream();
    };
  }, [showCameraStream, selectedCamera]);

  const startCameraStream = async () => {
    setErrorMessage(null);
    try {
      const constraints: MediaStreamConstraints = {
        video: {
          deviceId: selectedCamera ? { exact: selectedCamera } : undefined,
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      setCameraActive(true);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().catch(err => {
            console.error("Error playing video:", err);
            setErrorMessage("Failed to play camera stream");
          });
        };
      }
    } catch (err: any) {
      console.error("Error accessing camera:", err);
      let errorMsg = "Failed to access camera. ";
      
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        errorMsg += "Camera permission was denied. Please grant camera permissions.";
      } else if (err.name === "NotFoundError") {
        errorMsg += "No camera found. Please select a different camera.";
      } else if (err.name === "NotReadableError") {
        errorMsg += "Camera is already in use by another application.";
      } else {
        errorMsg += err.message;
      }
      
      setErrorMessage(errorMsg);
      setShowCameraStream(false);
      setCameraActive(false);
    }
  };

  const stopCameraStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        track.stop();
      });
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    setCameraActive(false);
  };

  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current) {
      setErrorMessage("Camera or canvas not available");
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (!context) {
      setErrorMessage("Could not get canvas context");
      return;
    }

    // Ensure video is ready
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      setErrorMessage("Camera feed is not ready yet");
      return;
    }

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Get the image data URL
    const imageData = canvas.toDataURL('image/jpeg', 0.9);
    setCapturedImageSrc(imageData);
    setShowCameraStream(false);
    setErrorMessage(null);
    
    // Store image dimensions
    setImageDimensions({ width: canvas.width, height: canvas.height });
    
    // Calculate camera scale for matching display
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    const container = document.querySelector('.camera-container');
    if (container) {
      const rect = container.getBoundingClientRect();
      const scaleX = videoWidth / rect.width;
      const scaleY = videoHeight / rect.height;
      setCameraScale({ scaleX, scaleY });
    }
  };

  const runEdgeDetection = () => {
  if (!capturedImageSrc) {
    setErrorMessage("No image captured for edge detection");
    return;
  }

  setIsDetectingEdges(true);
  setErrorMessage(null);

  const img = new Image();
  img.onload = () => {
    try {
      const canvas = document.createElement('canvas');
      const maxDimension = 800; // Limit size for performance
      let width = img.width;
      let height = img.height;
      
      // Scale down large images for better performance
      if (width > maxDimension || height > maxDimension) {
        const scale = Math.min(maxDimension / width, maxDimension / height);
        width = Math.floor(width * scale);
        height = Math.floor(height * scale);
        
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        if (!tempCtx) {
          throw new Error("Failed to create temporary canvas context");
        }
        
        tempCanvas.width = width;
        tempCanvas.height = height;
        tempCtx.drawImage(img, 0, 0, width, height);
        
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          throw new Error("Failed to get canvas context for edge detection");
        }
        
        ctx.drawImage(tempCanvas, 0, 0, width, height);
      } else {
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          throw new Error("Failed to get canvas context for edge detection");
        }
        ctx.drawImage(img, 0, 0, width, height);
      }
      
      // Get image data for edge detection
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error("Failed to get canvas context");
      }
      
      const imageData = ctx.getImageData(0, 0, width, height);
      
      console.log("Starting edge detection on image:", width, "x", height);
      
      // Perform edge detection with timeout
      setTimeout(() => {
        try {
          const { points: detectedPoints, edges } = detectEdges(imageData);
          console.log("Edge detection completed, found", detectedPoints.length, "points");
          
          if (detectedPoints.length >= 3) {
            setPolygonPoints(detectedPoints);
            drawEdges(edges, width, height, detectedPoints);
            setIsDetectingEdges(false);
            setEdgeDetectionMode(true);
          } else {
            throw new Error("Not enough edge points detected");
          }
        } catch (edgeError) {
          console.error("Error in edge detection:", edgeError);
          setErrorMessage("Edge detection failed: " + edgeError.message);
          setIsDetectingEdges(false);
        }
      }, 100);
      
    } catch (error: any) {
      console.error("Error in edge detection setup:", error);
      setErrorMessage("Failed to process image: " + error.message);
      setIsDetectingEdges(false);
    }
  };
  
  img.onerror = () => {
    setErrorMessage("Failed to load image for edge detection");
    setIsDetectingEdges(false);
  };
  
  img.src = capturedImageSrc;
};

  const drawEdges = (edges: boolean[][], width: number, height: number, points: {x: number; y: number}[]) => {
    if (!edgeCanvasRef.current) return;
    
    const canvas = edgeCanvasRef.current;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return;
    
    // Set canvas dimensions
    canvas.width = width;
    canvas.height = height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Create edge visualization
    const edgeImageData = ctx.createImageData(width, height);
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const index = (y * width + x) * 4;
        
        if (edges[y] && edges[y][x]) {
          // Draw edges in red
          edgeImageData.data[index] = 255;     // R
          edgeImageData.data[index + 1] = 0;   // G
          edgeImageData.data[index + 2] = 0;   // B
          edgeImageData.data[index + 3] = 255; // A
        } else {
          // Set pixel to original color
          edgeImageData.data[index] = 255;     // R
          edgeImageData.data[index + 1] = 255; // G
          edgeImageData.data[index + 2] = 255; // B
          edgeImageData.data[index + 3] = 50;  // A (semi-transparent)
        }
      }
    }
    
    // Put edge visualization on canvas
    ctx.putImageData(edgeImageData, 0, 0);
    
    // Draw polygon overlay
    ctx.beginPath();
    const pixelPoints = points.map(point => ({
      x: point.x * width,
      y: point.y * height
    }));
    
    ctx.moveTo(pixelPoints[0].x, pixelPoints[0].y);
    for (let i = 1; i < pixelPoints.length; i++) {
      ctx.lineTo(pixelPoints[i].x, pixelPoints[i].y);
    }
    ctx.closePath();
    
    ctx.strokeStyle = 'rgb(0, 255, 0)';
    ctx.lineWidth = 3;
    ctx.stroke();
    
    ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
    ctx.fill();
    
    // Draw vertices
    pixelPoints.forEach((point, index) => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = 'rgb(0, 255, 0)';
      ctx.fill();
      ctx.strokeStyle = 'rgb(0, 0, 0)';
      ctx.lineWidth = 2;
      ctx.stroke();
    });
  };

  const acceptDetectedEdges = () => {
    setEdgeDetectionMode(false);
    if (capturedImageSrc) {
      processAndAnalyzeImage();
    }
  };

  const calculateMeasurements = (points: {x: number; y: number}[], imageWidth: number, imageHeight: number) => {
    const basePixelToUnitFactor = 0.02; // cm per pixel at reference distance
    const referenceDistance = 10; // cm
    const scaleFactor = basePixelToUnitFactor * (cameraDistance / referenceDistance);

    const calculatedSideLengths: number[] = [];
    let totalPerimeter = 0;

    // Calculate actual pixel positions
    const pixelPoints = points.map(point => ({
      x: point.x * imageWidth,
      y: point.y * imageHeight
    }));

    for (let i = 0; i < pixelPoints.length; i++) {
      const p1 = pixelPoints[i];
      const p2 = pixelPoints[(i + 1) % pixelPoints.length];
      
      const pixelDistance = getDistance(p1, p2);
      const actualSideLength = pixelDistance * scaleFactor;
      const roundedLength = parseFloat(actualSideLength.toFixed(1));
      
      calculatedSideLengths.push(roundedLength);
      totalPerimeter += actualSideLength;
    }

    // Calculate area
    const pixelArea = getPolygonArea(pixelPoints);
    const calculatedArea = pixelArea * (scaleFactor * scaleFactor);
    const areaDisplay = parseFloat(calculatedArea.toFixed(2));
    const perimeterDisplay = parseFloat(totalPerimeter.toFixed(2));

    return { areaDisplay, perimeterDisplay, calculatedSideLengths, pixelPoints };
  };

  const drawProcessedImage = (points: {x: number; y: number}[]) => {
    if (!capturedImageSrc || !overlayCanvasRef.current) {
      setErrorMessage("No image captured to process");
      return;
    }

    const img = new Image();
    img.onload = () => {
      const canvas = overlayCanvasRef.current!;
      const ctx = canvas.getContext('2d')!;
      
      // Set canvas dimensions
      canvas.width = img.width;
      canvas.height = img.height;

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw the captured image
      ctx.drawImage(img, 0, 0, img.width, img.height);

      // Calculate actual pixel positions
      const pixelPoints = points.map(point => ({
        x: point.x * img.width,
        y: point.y * img.height
      }));

      // Draw the polygon outline
      ctx.beginPath();
      ctx.moveTo(pixelPoints[0].x, pixelPoints[0].y);
      for (let i = 1; i < pixelPoints.length; i++) {
        ctx.lineTo(pixelPoints[i].x, pixelPoints[i].y);
      }
      ctx.closePath();
      
      // Stroke style
      ctx.strokeStyle = 'rgb(59, 130, 246)';
      ctx.lineWidth = 4;
      ctx.stroke();
      
      // Fill with semi-transparent color
      ctx.fillStyle = 'rgba(59, 130, 246, 0.2)';
      ctx.fill();

      // Calculate and draw measurements
      const { areaDisplay, perimeterDisplay, calculatedSideLengths } = calculateMeasurements(points, img.width, img.height);

      // Draw side lengths
      ctx.font = 'bold 20px Arial';
      ctx.fillStyle = 'rgb(239, 68, 68)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      for (let i = 0; i < pixelPoints.length; i++) {
        const p1 = pixelPoints[i];
        const p2 = pixelPoints[(i + 1) % pixelPoints.length];
        
        const midX = (p1.x + p2.x) / 2;
        const midY = (p1.y + p2.y) / 2;
        
        // Add background for better readability
        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.fillRect(midX - 50, midY - 20, 100, 30);
        
        ctx.fillStyle = 'rgb(239, 68, 68)';
        ctx.fillText(`${calculatedSideLengths[i]} cm`, midX, midY - 5);
      }

      // Draw area and perimeter
      ctx.font = 'bold 24px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      // Add background for area text
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.fillRect(img.width / 2 - 150, img.height / 2 - 60, 300, 120);
      
      ctx.fillStyle = 'rgb(16, 185, 129)';
      ctx.fillText(`Area: ${areaDisplay} cm²`, img.width / 2, img.height / 2 - 20);
      
      ctx.fillStyle = 'rgb(239, 68, 68)';
      ctx.fillText(`Perimeter: ${perimeterDisplay} cm`, img.width / 2, img.height / 2 + 20);

      // Draw vertices
      pixelPoints.forEach((point, index) => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = 'white';
        ctx.fill();
        ctx.strokeStyle = 'rgb(59, 130, 246)';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Vertex number
        ctx.fillStyle = 'rgb(59, 130, 246)';
        ctx.font = 'bold 16px Arial';
        ctx.fillText((index + 1).toString(), point.x, point.y - 20);
      });

      // Set measurements state
      setMeasurements({
        area: areaDisplay,
        perimeter: perimeterDisplay,
        sideLengths: calculatedSideLengths,
      });

      // Set processed image
      setProcessedImageSrc(canvas.toDataURL('image/jpeg', 0.9));
    };

    img.onerror = () => {
      setErrorMessage("Failed to load image for processing");
    };

    img.src = capturedImageSrc;
  };

  const processAndAnalyzeImage = () => {
    if (!capturedImageSrc) {
      setErrorMessage("No image captured to process");
      return;
    }

    setIsProcessing(true);
    setErrorMessage(null);

    // Draw with current polygon points
    drawProcessedImage(polygonPoints);
    
    setTimeout(() => {
      setIsProcessing(false);
    }, 1000);
  };

  const handleEditMode = () => {
    setEditMode(true);
  };

  const saveEdits = () => {
    setEditMode(false);
    // Recalculate measurements with edited points
    if (capturedImageSrc && imageDimensions.width > 0 && imageDimensions.height > 0) {
      const { areaDisplay, perimeterDisplay, calculatedSideLengths } = calculateMeasurements(
        polygonPoints, 
        imageDimensions.width, 
        imageDimensions.height
      );
      
      setMeasurements({
        area: areaDisplay,
        perimeter: perimeterDisplay,
        sideLengths: calculatedSideLengths,
      });
      
      // Redraw the processed image
      drawProcessedImage(polygonPoints);
    }
  };

  const cancelEdits = () => {
    setEditMode(false);
  };

  const handleEditCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!editMode || !capturedImageSrc) return;

    const canvas = editCanvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Scale coordinates to image space
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const scaledX = x * scaleX;
    const scaledY = y * scaleY;

    // Check if clicked near a point
    for (let i = 0; i < polygonPoints.length; i++) {
      const point = polygonPoints[i];
      const pointX = point.x * canvas.width;
      const pointY = point.y * canvas.height;
      const distance = Math.sqrt((scaledX - pointX) ** 2 + (scaledY - pointY) ** 2);
      
      if (distance < 20) { // 20 pixel radius for click detection
        setActivePointIndex(i);
        return;
      }
    }
  };

  const handleEditCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!editMode || activePointIndex === null || !capturedImageSrc) return;

    const canvas = editCanvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Scale coordinates to image space
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const scaledX = x * scaleX;
    const scaledY = y * scaleY;

    // Update the point position
    const updatedPoints = [...polygonPoints];
    updatedPoints[activePointIndex] = {
      x: scaledX / canvas.width,
      y: scaledY / canvas.height
    };
    
    setPolygonPoints(updatedPoints);
    
    // Redraw edit canvas
    drawEditCanvas(updatedPoints);
  };

  const handleEditCanvasMouseUp = () => {
    setActivePointIndex(null);
  };

  const handleEditCanvasDoubleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!editMode || !capturedImageSrc) return;

    const canvas = editCanvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Scale coordinates to image space
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const scaledX = x * scaleX;
    const scaledY = y * scaleY;

    // Check if double-clicked near a point
    for (let i = 0; i < polygonPoints.length; i++) {
      const point = polygonPoints[i];
      const pointX = point.x * canvas.width;
      const pointY = point.y * canvas.height;
      const distance = Math.sqrt((scaledX - pointX) ** 2 + (scaledY - pointY) ** 2);
      
      if (distance < 20) { // 20 pixel radius for double-click detection
        saveEdits();
        return;
      }
    }
  };

  const drawEditCanvas = (points: {x: number; y: number}[]) => {
    if (!capturedImageSrc || !editCanvasRef.current) return;

    const img = new Image();
    img.onload = () => {
      const canvas = editCanvasRef.current!;
      const ctx = canvas.getContext('2d')!;
      
      // Set canvas dimensions
      canvas.width = img.width;
      canvas.height = img.height;

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw the captured image
      ctx.drawImage(img, 0, 0, img.width, img.height);

      // Calculate actual pixel positions
      const pixelPoints = points.map(point => ({
        x: point.x * img.width,
        y: point.y * img.height
      }));

      // Draw the polygon outline
      ctx.beginPath();
      ctx.moveTo(pixelPoints[0].x, pixelPoints[0].y);
      for (let i = 1; i < pixelPoints.length; i++) {
        ctx.lineTo(pixelPoints[i].x, pixelPoints[i].y);
      }
      ctx.closePath();
      
      // Stroke style
      ctx.strokeStyle = 'rgb(59, 130, 246)';
      ctx.lineWidth = 4;
      ctx.stroke();
      
      // Fill with semi-transparent color
      ctx.fillStyle = 'rgba(59, 130, 246, 0.2)';
      ctx.fill();

      // Draw vertices as draggable handles
      pixelPoints.forEach((point, index) => {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 12, 0, Math.PI * 2);
        ctx.fillStyle = index === activePointIndex ? 'rgb(239, 68, 68)' : 'white';
        ctx.fill();
        ctx.strokeStyle = index === activePointIndex ? 'rgb(239, 68, 68)' : 'rgb(59, 130, 246)';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Vertex number
        ctx.fillStyle = 'rgb(59, 130, 246)';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText((index + 1).toString(), point.x, point.y);
      });

      // Draw instruction text
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(20, 20, 400, 60);
      ctx.fillStyle = 'white';
      ctx.font = 'bold 16px Arial';
      ctx.textAlign = 'left';
      ctx.fillText('Drag the circles to adjust the outline', 30, 40);
      ctx.fillText('Double-click on a circle to save and exit edit mode', 30, 65);
    };

    img.src = capturedImageSrc;
  };

  useEffect(() => {
    if (editMode && capturedImageSrc) {
      drawEditCanvas(polygonPoints);
    }
  }, [editMode, capturedImageSrc, polygonPoints]);

  const resetApp = () => {
    stopCameraStream();
    setShowCameraStream(false);
    setCameraActive(false);
    setCapturedImageSrc(null);
    setProcessedImageSrc(null);
    setMeasurements({ area: null, perimeter: null, sideLengths: null });
    setErrorMessage(null);
    setIsProcessing(false);
    setEditMode(false);
    setEdgeDetectionMode(false);
    setIsDetectingEdges(false);
  };

  const handleCameraChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedCamera(e.target.value);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 p-4 md:p-6 flex items-center justify-center font-sans text-gray-800">
      <div className="bg-white p-4 md:p-8 rounded-3xl shadow-xl w-full max-w-6xl border border-gray-200">
        <h1 className="text-3xl md:text-4xl font-extrabold text-center mb-6 md:mb-8 text-blue-700">
          Paper Pattern Scanner
        </h1>

        {errorMessage && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-xl mb-6" role="alert">
            <strong className="font-bold">Error!</strong>
            <span className="ml-2">{errorMessage}</span>
          </div>
        )}

        {/* Camera Selection and Controls */}
        <div className="mb-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            {availableCameras.length > 0 && (
              <div>
                <label className="block text-gray-700 text-sm font-bold mb-2">
                  Select Camera
                </label>
                <select
                  value={selectedCamera}
                  onChange={handleCameraChange}
                  className="w-full px-4 py-3 rounded-lg border-2 border-blue-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition duration-200 text-lg"
                  disabled={cameraActive}
                >
                  {availableCameras.map((camera) => (
                    <option key={camera.deviceId} value={camera.deviceId}>
                      {camera.label || `Camera ${camera.deviceId.slice(0, 8)}`}
                    </option>
                  ))}
                </select>
              </div>
            )}
            
            <div>
              <label className="block text-gray-700 text-sm font-bold mb-2">
                Camera Distance: {cameraDistance} cm
              </label>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min="10"
                  max="100"
                  value={cameraDistance}
                  onChange={(e) => {
                    setCameraDistance(parseInt(e.target.value));
                    if (capturedImageSrc && !editMode && !edgeDetectionMode) {
                      processAndAnalyzeImage();
                    }
                  }}
                  className="flex-1 h-2 bg-blue-100 rounded-lg appearance-none cursor-pointer"
                  disabled={isProcessing}
                />
                <input
                  type="number"
                  value={cameraDistance}
                  onChange={(e) => {
                    const value = parseInt(e.target.value);
                    if (!isNaN(value) && value >= 10 && value <= 100) {
                      setCameraDistance(value);
                      if (capturedImageSrc && !editMode && !edgeDetectionMode) {
                        setTimeout(() => processAndAnalyzeImage(), 100);
                      }
                    }
                  }}
                  className="w-24 px-3 py-2 rounded-lg border-2 border-blue-300 focus:border-blue-500 text-center"
                  min="10"
                  max="100"
                  disabled={isProcessing}
                />
              </div>
            </div>
          </div>

          <div className="flex flex-wrap justify-center gap-4 mb-8">
            <button
              onClick={showCameraStream ? captureImage : () => setShowCameraStream(true)}
              className={`flex items-center justify-center gap-2 px-6 py-3 rounded-full text-white text-lg font-semibold transition duration-300
                ${showCameraStream 
                  ? 'bg-blue-600 hover:bg-blue-700' 
                  : 'bg-green-600 hover:bg-green-700'
                } shadow-lg`}
              disabled={isProcessing || (showCameraStream && !cameraActive)}
            >
              {showCameraStream ? (
                <>
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.86-1.291A2 2 0 0110.451 4h3.098a2 2 0 011.664.89l.86 1.291A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  Capture Image
                </>
              ) : (
                <>
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.86-1.291A2 2 0 0110.451 4h3.098a2 2 0 011.664.89l.86 1.291A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  Start Camera
                </>
              )}
            </button>

            {capturedImageSrc && !showCameraStream && !processedImageSrc && !editMode && !edgeDetectionMode && (
              <>
                <button
                  onClick={runEdgeDetection}
                  className="flex items-center justify-center gap-2 px-6 py-3 rounded-full bg-purple-600 hover:bg-purple-700 text-white text-lg font-semibold transition duration-300 shadow-lg"
                  disabled={isProcessing || isDetectingEdges}
                >
                  {isDetectingEdges ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-white"></div>
                      Detecting Edges...
                    </>
                  ) : (
                    <>
                      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6a2 2 0 00-2-2H5a2 2 0 00-2 2v13a2 2 0 002 2h4zm0 0h10a2 2 0 002-2V6a2 2 0 00-2-2H9m11 11l-3-3m0 0l-3 3m3-3v7" />
                      </svg>
                      Auto-Detect Outline
                    </>
                  )}
                </button>
                <button
                  onClick={processAndAnalyzeImage}
                  className="flex items-center justify-center gap-2 px-6 py-3 rounded-full bg-teal-600 hover:bg-teal-700 text-white text-lg font-semibold transition duration-300 shadow-lg"
                  disabled={isProcessing}
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6a2 2 0 00-2-2H5a2 2 0 00-2 2v13a2 2 0 002 2h4zm0 0h10a2 2 0 002-2V6a2 2 0 00-2-2H9m11 11l-3-3m0 0l-3 3m3-3v7" />
                  </svg>
                  Analyze with Current Outline
                </button>
              </>
            )}

            {processedImageSrc && !editMode && !edgeDetectionMode && (
              <button
                onClick={handleEditMode}
                className="flex items-center justify-center gap-2 px-6 py-3 rounded-full bg-amber-500 hover:bg-amber-600 text-white text-lg font-semibold transition duration-300 shadow-lg"
                disabled={isProcessing}
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
                Edit Outline
              </button>
            )}

            {edgeDetectionMode && (
              <button
                onClick={acceptDetectedEdges}
                className="flex items-center justify-center gap-2 px-6 py-3 rounded-full bg-green-600 hover:bg-green-700 text-white text-lg font-semibold transition duration-300 shadow-lg"
                disabled={isProcessing}
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                Use Detected Outline
              </button>
            )}

            {editMode && (
              <>
                <button
                  onClick={saveEdits}
                  className="flex items-center justify-center gap-2 px-6 py-3 rounded-full bg-green-600 hover:bg-green-700 text-white text-lg font-semibold transition duration-300 shadow-lg"
                  disabled={isProcessing}
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Save Changes
                </button>
                <button
                  onClick={cancelEdits}
                  className="flex items-center justify-center gap-2 px-6 py-3 rounded-full bg-gray-600 hover:bg-gray-700 text-white text-lg font-semibold transition duration-300 shadow-lg"
                  disabled={isProcessing}
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                  Cancel
                </button>
              </>
            )}

            <button
              onClick={resetApp}
              className="flex items-center justify-center gap-2 px-6 py-3 rounded-full bg-red-600 hover:bg-red-700 text-white text-lg font-semibold transition duration-300 shadow-lg"
              disabled={isProcessing}
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Reset
            </button>
          </div>
        </div>

        {/* Camera Stream / Captured Image */}
        <div className="relative bg-gray-100 rounded-2xl overflow-hidden shadow-inner border border-gray-300 mb-8 aspect-video flex items-center justify-center camera-container">
          {showCameraStream && (
            <>
              <video 
                ref={videoRef} 
                autoPlay 
                playsInline 
                muted 
                className="absolute inset-0 w-full h-full object-contain"
                style={{
                  transform: `scale(${cameraScale.scaleX}, ${cameraScale.scaleY})`,
                  transformOrigin: 'center'
                }}
              />
              {!cameraActive && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-50">
                  <div className="text-white text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mx-auto mb-4"></div>
                    <p className="text-xl font-semibold">Starting camera...</p>
                  </div>
                </div>
              )}
            </>
          )}

          {!showCameraStream && !capturedImageSrc && !processedImageSrc && !editMode && !edgeDetectionMode && (
            <div className="flex flex-col items-center text-gray-400 p-8 text-center">
              <svg className="w-24 h-24 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p className="text-xl font-medium">Click "Start Camera" to begin scanning.</p>
            </div>
          )}

          {!showCameraStream && capturedImageSrc && !processedImageSrc && !editMode && !edgeDetectionMode && (
            <img 
              src={capturedImageSrc} 
              alt="Captured Paper Pattern" 
              className="max-h-full max-w-full object-contain rounded-2xl"
            />
          )}

          {!showCameraStream && processedImageSrc && !editMode && !edgeDetectionMode && (
            <img 
              src={processedImageSrc} 
              alt="Processed Pattern with Measurements" 
              className="max-h-full max-w-full object-contain rounded-2xl"
            />
          )}

          {editMode && capturedImageSrc && (
            <canvas
              ref={editCanvasRef}
              className="max-h-full max-w-full object-contain rounded-2xl cursor-move"
              onClick={handleEditCanvasClick}
              onDoubleClick={handleEditCanvasDoubleClick}
              onMouseMove={handleEditCanvasMouseMove}
              onMouseUp={handleEditCanvasMouseUp}
              onMouseLeave={handleEditCanvasMouseUp}
            />
          )}

          {edgeDetectionMode && capturedImageSrc && (
            <canvas
              ref={edgeCanvasRef}
              className="max-h-full max-w-full object-contain rounded-2xl"
            />
          )}

          {isProcessing && (
            <div className="absolute inset-0 bg-gray-900 bg-opacity-75 flex flex-col items-center justify-center text-white text-xl font-semibold z-10 rounded-2xl">
              <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-blue-500 mb-4"></div>
              Processing Image...
            </div>
          )}

          {isDetectingEdges && (
            <div className="absolute inset-0 bg-gray-900 bg-opacity-75 flex flex-col items-center justify-center text-white text-xl font-semibold z-10 rounded-2xl">
              <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-purple-500 mb-4"></div>
              Detecting edges...
            </div>
          )}
        </div>

        {/* Measurement Results */}
        {measurements.area !== null && !editMode && !edgeDetectionMode && (
          <div className="bg-gradient-to-r from-blue-50 to-emerald-50 border border-blue-200 p-6 rounded-2xl shadow-md">
            <h2 className="text-2xl md:text-3xl font-bold text-blue-800 mb-4 text-center">Analysis Results</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white p-6 rounded-xl shadow-sm border border-blue-100 text-center">
                <p className="text-gray-600 font-medium text-lg mb-2">Area</p>
                <p className="text-blue-700 text-3xl md:text-4xl font-extrabold">{measurements.area} cm²</p>
              </div>
              <div className="bg-white p-6 rounded-xl shadow-sm border border-blue-100 text-center">
                <p className="text-gray-600 font-medium text-lg mb-2">Perimeter</p>
                <p className="text-emerald-700 text-3xl md:text-4xl font-extrabold">{measurements.perimeter} cm</p>
              </div>
              <div className="bg-white p-6 rounded-xl shadow-sm border border-blue-100 text-center">
                <p className="text-gray-600 font-medium text-lg mb-2">Average Side Length</p>
                <p className="text-purple-700 text-3xl md:text-4xl font-extrabold">
                  {measurements.sideLengths 
                    ? `${(measurements.sideLengths.reduce((a, b) => a + b, 0) / measurements.sideLengths.length).toFixed(1)} cm`
                    : 'N/A'
                  }
                </p>
              </div>
            </div>
            
            {measurements.sideLengths && (
              <div className="mt-6">
                <p className="text-gray-600 font-medium text-lg mb-3 text-center">Individual Side Lengths</p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {measurements.sideLengths.map((length, index) => (
                    <div key={index} className="bg-blue-50 p-3 rounded-lg text-center">
                      <p className="text-gray-600 text-sm">Side {index + 1}</p>
                      <p className="text-blue-700 text-lg font-semibold">{length.toFixed(1)} cm</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Hidden canvases */}
        <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>
        <canvas ref={overlayCanvasRef} style={{ display: 'none' }}></canvas>
        <canvas ref={edgeCanvasRef} style={{ display: 'none' }}></canvas>
      </div>
    </div>
  );
}
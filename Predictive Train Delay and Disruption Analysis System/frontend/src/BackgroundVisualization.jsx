import { useEffect, useRef } from 'react';
import * as THREE from 'three';

export default function BackgroundVisualization() {
  const mountRef = useRef(null);
  const animationRef = useRef(null);

  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.1, 10);
    camera.position.z = 1.5;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x000000, 0); // Transparent background
    mountRef.current.appendChild(renderer.domElement);

    // Lights
    const sphereGeometry = new THREE.SphereGeometry(0.01, 16, 8);
    
    const light1 = new THREE.PointLight(0x3b82f6, 0.3, 2); // Blue
    const light2 = new THREE.PointLight(0x60a5fa, 0.3, 2); // Light blue
    const light3 = new THREE.PointLight(0x1d4ed8, 0.3, 2); // Dark blue

    // Light indicators (smaller and more subtle)
    const lightMaterial1 = new THREE.MeshBasicMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.6 });
    const lightMaterial2 = new THREE.MeshBasicMaterial({ color: 0x60a5fa, transparent: true, opacity: 0.6 });
    const lightMaterial3 = new THREE.MeshBasicMaterial({ color: 0x1d4ed8, transparent: true, opacity: 0.6 });

    const lightMesh1 = new THREE.Mesh(sphereGeometry, lightMaterial1);
    const lightMesh2 = new THREE.Mesh(sphereGeometry, lightMaterial2);
    const lightMesh3 = new THREE.Mesh(sphereGeometry, lightMaterial3);

    light1.add(lightMesh1);
    light2.add(lightMesh2);
    light3.add(lightMesh3);

    scene.add(light1, light2, light3);

    // Points (fewer for better performance as background)
    const points = [];
    for (let i = 0; i < 20000; i++) {
      const point = new THREE.Vector3()
        .random()
        .subScalar(0.5)
        .multiplyScalar(3);
      points.push(point);
    }

    const geometryPoints = new THREE.BufferGeometry().setFromPoints(points);
    const materialPoints = new THREE.PointsMaterial({
      color: 0x94a3b8,
      size: 0.005,
      transparent: true,
      opacity: 0.3
    });

    const pointCloud = new THREE.Points(geometryPoints, materialPoints);
    scene.add(pointCloud);

    // Animation
    const animate = () => {
      const time = Date.now() * 0.0005; // Slower animation
      const scale = 0.8;

      light1.position.x = Math.sin(time * 0.7) * scale;
      light1.position.y = Math.cos(time * 0.5) * scale;
      light1.position.z = Math.cos(time * 0.3) * scale;

      light2.position.x = Math.cos(time * 0.3) * scale;
      light2.position.y = Math.sin(time * 0.5) * scale;
      light2.position.z = Math.sin(time * 0.7) * scale;

      light3.position.x = Math.sin(time * 0.7) * scale;
      light3.position.y = Math.cos(time * 0.3) * scale;
      light3.position.z = Math.sin(time * 0.5) * scale;

      scene.rotation.y = time * 0.05; // Slower rotation

      renderer.render(scene, camera);
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    // Handle window resize
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  return (
    <div 
      ref={mountRef} 
      style={{ 
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        zIndex: -1,
        pointerEvents: 'none'
      }} 
    />
  );
}
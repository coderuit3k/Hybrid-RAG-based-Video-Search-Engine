import { useState } from 'react'
import { FiImage, FiX, FiInfo, FiTag, FiClock } from 'react-icons/fi'

function ResultsGrid({ results }) {
    const [hoveredIndex, setHoveredIndex] = useState(null)
    const [selectedResult, setSelectedResult] = useState(null)

    // Helper to construct image URL
    const getImageUrl = (result) => {
        if (!result.image_path) return '';
        
        // If it's already a URL (S3 or other), use it directly
        if (result.image_path.startsWith('http') || result.image_path.startsWith('https') || result.image_path.startsWith('s3://')) {
            return result.image_path;
        }

        // Legacy local path fallback
        const filename = result.image_path.split(/[/\\]/).pop();
        if (result.video_id) {
            const libraryId = result.video_id.split('_')[0];
            return `/static/keyframes/${libraryId}/${result.video_id}/${filename}`;
        }
        return `/static/keyframes/${filename}`;
    };

    // Helper to format frame index to timestamp (FPS=25)
    // Returns HH:MM:SS
    const formatTimestamp = (frameIdx) => {
        if (frameIdx === undefined || frameIdx === null) return '';
        const fps = 25;
        const totalSeconds = Math.floor(frameIdx / fps);
        
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;

        if (hours > 0) {
            return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
        return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    };

    // Helper to construct video URL
    const getVideoUrl = (videoId) => {
        if (!videoId) return '';
        // Construct S3 URL based on user config
        const bucket = "video-search-bucket-thanh-2025";
        const region = "ap-southeast-2";
        return `https://${bucket}.s3.${region}.amazonaws.com/videos/${videoId}.mp4`;
    };

    return (
        <div className="mt-12">
            <h2 className="text-3xl font-bold mb-6 gradient-text">
                Search Results ({results.length})
            </h2>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {results.map((result, index) => (
                    <div
                        key={index}
                        className="relative group cursor-pointer overflow-hidden rounded-xl glass-effect border-2 border-dark-border hover:border-primary transition-all duration-300 transform hover:scale-105"
                        onMouseEnter={() => setHoveredIndex(index)}
                        onMouseLeave={() => setHoveredIndex(null)}
                        onClick={() => setSelectedResult(result)}
                    >
                        {/* Image */}
                        <div className="aspect-video bg-dark-surface flex items-center justify-center overflow-hidden">
                            {result.image_path ? (
                                <img
                                    src={getImageUrl(result)}
                                    alt={`Frame ${result.frame_idx}`}
                                    className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
                                    onError={(e) => {
                                        e.target.onerror = null;
                                        if (!e.target.dataset.retry) {
                                            e.target.dataset.retry = "true";
                                            // Handle fallback if needed
                                        }
                                        e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iIzFhMjAzMCIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTYiIGZpbGw9IiM2Mzc2OTEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5JbWFnZSBub3QgZm91bmQ8L3RleHQ+PC9zdmc+'
                                    }}
                                />
                            ) : (
                                <FiImage className="text-6xl text-gray-600" />
                            )}
                            
                            {/* Timestamp Badge */}
                            <div className="absolute top-2 right-2 bg-black/70 backdrop-blur-sm px-2 py-0.5 rounded text-xs font-mono text-white flex items-center gap-1">
                                <FiClock size={10} />
                                {formatTimestamp(result.frame_idx)}
                            </div>
                        </div>

                        {/* Metadata Overlay */}
                        <div
                            className={`absolute inset-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent flex flex-col justify-end p-4 transition-opacity duration-300 ${hoveredIndex === index ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
                                }`}
                        >
                            <p className="text-sm font-semibold text-primary mb-1">
                                Video: {result.video_id}
                            </p>
                            <p className="text-sm text-gray-300 mb-1 flex justify-between">
                                <span>Frame: {result.frame_idx}</span>
                            </p>
                            <div className="flex items-center gap-2">
                                <div className="h-2 flex-1 bg-dark-surface rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-gradient-to-r from-primary to-secondary rounded-full"
                                        style={{ width: `${Math.min(result.score * 100, 100)}%` }}
                                    ></div>
                                </div>
                                <span className="text-xs font-bold text-primary">
                                    {(result.score * 100).toFixed(1)}%
                                </span>
                            </div>
                            {result.ocr_text && (
                                <p className="text-xs text-gray-400 mt-2 truncate">
                                    OCR: {result.ocr_text}
                                </p>
                            )}
                        </div>

                        {/* Rank Badge */}
                        <div className="absolute top-2 left-2 bg-gradient-to-r from-primary to-secondary px-3 py-1 rounded-full text-xs font-bold text-white shadow-lg">
                            #{index + 1}
                        </div>
                    </div>
                ))}
            </div>

            {/* Image Zoom Modal */}
            {selectedResult && (
                <div 
                    className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-md animate-fade-in"
                    onClick={() => setSelectedResult(null)}
                >
                    <div 
                        className="relative max-w-7xl w-full max-h-[90vh] bg-dark-surface border border-gray-700/50 rounded-2xl overflow-hidden flex flex-col md:flex-row shadow-2xl animate-scale-in"
                        onClick={(e) => e.stopPropagation()} // Prevent closing when clicking modal content
                    >
                        {/* Close Button */}
                        <button 
                            onClick={() => setSelectedResult(null)}
                            className="absolute top-4 right-4 z-10 p-2 bg-black/50 hover:bg-red-500/80 rounded-full text-white transition-colors"
                        >
                            <FiX size={24} />
                        </button>

                        {/* Main Content (Video or Image) */}
                        <div className="flex-1 bg-black flex items-center justify-center p-2 relative h-[50vh] md:h-auto group">
                           {/* Video Player */}
                           <video
                                src={`${getVideoUrl(selectedResult.video_id)}#t=${selectedResult.frame_idx / 25}`}
                                controls
                                autoPlay
                                className="max-w-full max-h-full object-contain"
                                onError={(e) => {
                                    // Fallback to image if video fails
                                    e.target.style.display = 'none';
                                    e.target.nextElementSibling.style.display = 'block';
                                }}
                           />
                           
                           {/* Fallback Image (hidden by default if video loads) */}
                           <img
                                src={getImageUrl(selectedResult)}
                                alt={`Frame ${selectedResult.frame_idx}`}
                                className="max-w-full max-h-full object-contain hidden"
                                style={{ display: 'none' }} 
                            />
                        </div>

                        {/* Sidebar Info */}
                        <div className="w-full md:w-96 bg-dark-bg/95 backdrop-blur border-l border-gray-700/50 p-6 flex flex-col overflow-y-auto">
                            <h3 className="text-2xl font-bold gradient-text mb-6">
                                Details
                            </h3>
                            
                            <div className="space-y-6">
                                <div className="space-y-2">
                                    <div className="flex items-center gap-2 text-gray-400">
                                        <FiImage className="text-primary" />
                                        <span className="text-sm uppercase tracking-wider">Video Source</span>
                                    </div>
                                    <p className="text-lg font-mono text-white break-all">
                                        {selectedResult.video_id}
                                    </p>
                                </div>

                                <div className="space-y-2">
                                    <div className="flex items-center gap-2 text-gray-400">
                                        <FiClock className="text-secondary" />
                                        <span className="text-sm uppercase tracking-wider">Time & Frame</span>
                                    </div>
                                    <p className="text-3xl text-white font-mono font-bold">
                                        {formatTimestamp(selectedResult.frame_idx)}
                                    </p>
                                    <p className="text-sm text-gray-500 font-mono">
                                        Frame Index: {selectedResult.frame_idx}
                                    </p>
                                </div>

                                <div className="space-y-2">
                                    <div className="flex items-center gap-2 text-gray-400">
                                        <FiTag className="text-green-500" />
                                        <span className="text-sm uppercase tracking-wider">Relevance Score</span>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <div className="flex-1 h-3 bg-dark-surface rounded-full overflow-hidden border border-gray-700">
                                            <div
                                                className="h-full bg-gradient-to-r from-primary to-secondary rounded-full"
                                                style={{ width: `${Math.min(selectedResult.score * 100, 100)}%` }}
                                            ></div>
                                        </div>
                                        <span className="text-xl font-bold text-white">
                                            {(selectedResult.score * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                </div>

                                {selectedResult.ocr_text && (
                                    <div className="space-y-2 p-4 bg-dark-surface/50 rounded-lg border border-gray-700/50">
                                        <div className="flex items-center gap-2 text-gray-400 mb-2">
                                            <FiInfo className="text-blue-400" />
                                            <span className="text-sm uppercase tracking-wider">OCR Text</span>
                                        </div>
                                        <p className="text-gray-300 text-sm italic leading-relaxed">
                                            "{selectedResult.ocr_text}"
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

export default ResultsGrid
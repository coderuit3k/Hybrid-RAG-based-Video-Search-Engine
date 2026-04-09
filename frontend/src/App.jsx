import { useState } from 'react'
import SearchInput from './components/SearchInput'
import ResultsGrid from './components/ResultsGrid'
import './index.css'

function App() {
    const [results, setResults] = useState([])
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    return (
        <div className="min-h-screen bg-gradient-to-br from-dark-bg via-dark-surface to-dark-bg">
            {/* Header */}
            <header className="py-8 px-4">
                <div className="container mx-auto max-w-6xl">
                    <h1 className="text-5xl font-bold text-center mb-2 gradient-text">
                        Multimodal Video Search
                    </h1>
                    <p className="text-center text-gray-400 text-lg">
                        Search videos using text with AI-powered hybrid retrieval
                    </p>
                </div>
            </header>

            {/* Main Content */}
            <main className="container mx-auto max-w-6xl px-4 pb-12">
                {/* Search Input */}
                <SearchInput
                    setResults={setResults}
                    setLoading={setLoading}
                    setError={setError}
                />

                {/* Error Display */}
                {error && (
                    <div className="mt-8 p-4 glass-effect border-l-4 border-red-500 text-red-300">
                        <p className="font-semibold">Error:</p>
                        <p>{error}</p>
                    </div>
                )}

                {/* Loading State */}
                {loading && (
                    <div className="mt-12 flex flex-col items-center justify-center">
                        <div className="spinner animate-glow"></div>
                        <p className="mt-4 text-gray-400 text-lg">
                            Searching...
                        </p>
                    </div>
                )}

                {/* Results Grid */}
                {!loading && results.length > 0 && (
                    <ResultsGrid results={results} />
                )}

                {/* Empty State */}
                {!loading && !error && results.length === 0 && (
                    <div className="mt-16 text-center">
                        <div className="inline-block p-6 glass-effect rounded-2xl">
                            <div className="text-6xl mb-4">🎬</div>
                            <p className="text-gray-400 text-lg">
                                Enter a search query to find similar keyframes
                            </p>
                        </div>
                    </div>
                )}
            </main>

            {/* Footer */}
            <footer className="py-6 text-center text-gray-500 border-t border-dark-border">
                <p>Powered by Milvus GPU + CLIP + Reranker</p>
            </footer>
        </div>
    )
}

export default App
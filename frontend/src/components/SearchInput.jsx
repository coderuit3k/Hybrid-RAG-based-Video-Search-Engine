import { useState } from 'react'
import axios from 'axios'
import { FiSend } from 'react-icons/fi'

function SearchInput({ setResults, setLoading, setError }) {
    const [query, setQuery] = useState('')

    const handleSearch = async (e) => {
        e.preventDefault()

        if (!query.trim()) {
            setError('Please enter a search query')
            return
        }

        setLoading(true)
        setError(null)
        setResults([])

        try {
            const endpoint = '/api/search/text'
            const payload = { query: query.trim() }

            const response = await axios.post(endpoint, payload)

            setResults(response.data.results || [])

            if (response.data.results.length === 0) {
                setError('No results found. Try a different query.')
            }
        } catch (err) {
            console.error('Search error:', err)
            setError(err.response?.data?.detail || err.message || 'Search failed. Please try again.')
        } finally {
            setLoading(false)
        }
    }

    return (
        <form onSubmit={handleSearch} className="glass-effect p-6 rounded-2xl shadow-2xl">
            <div className="flex flex-col md:flex-row gap-4">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Enter search query"
                    className="flex-1 px-6 py-4 bg-dark-surface border-2 border-dark-border rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/50 transition-all duration-300"
                />

                <button
                    type="submit"
                    className="px-8 py-4 gradient-button text-white font-semibold rounded-lg flex items-center justify-center gap-2 shadow-lg"
                >
                    <FiSend className="text-xl" />
                    Search
                </button>
            </div>
        </form>
    )
}

export default SearchInput
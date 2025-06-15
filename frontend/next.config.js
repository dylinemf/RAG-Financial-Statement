/** @type {import('next').NextConfig} */
const nextConfig = {
  
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*', // proxy dev request API Next.js ke FastAPI
      },
    ]
  },
  reactStrictMode: true,
  swcMinify: true,
}

module.exports = nextConfig 
import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'

// The public Vercel site has no backend, so it ships only the landing page
// (build with VITE_LANDING_ONLY=1). Dynamic imports keep the app's 3D/plot bundles out of that build.
async function main() {
  const root = ReactDOM.createRoot(document.getElementById('root')!)
  if (import.meta.env.VITE_LANDING_ONLY === '1') {
    const { default: LandingPage } = await import('./pages/LandingPage')
    root.render(<React.StrictMode><LandingPage /></React.StrictMode>)
    return
  }
  const { default: App } = await import('./App')
  root.render(
    <React.StrictMode>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </React.StrictMode>,
  )
}

void main()

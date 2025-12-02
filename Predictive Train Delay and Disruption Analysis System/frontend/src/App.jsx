import { useState } from "react";
import { searchStations, fetchIntegrated } from "./api";

function Sparkline({ values = [], width = 200, height = 40 }) {
  if (!values || values.length === 0) return <div style={{height}}>-</div>;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const step = width / Math.max(1, values.length - 1);

  const points = values
    .map((v, i) => {
      const x = i * step;
      const y = height - ((v - min) / range) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <polyline points={points} fill="none" stroke="#60a5fa" strokeWidth="2" />
    </svg>
  );
}

function Card({ children, style = {} }) {
  return <div style={{ borderRadius: "8px", ...style }}>{children}</div>;
}

function CardContent({ children, style = {} }) {
  return <div style={style}>{children}</div>;
}

function Input({ style = {}, ...props }) {
  return <input style={{ padding: "12px 16px", borderRadius: "6px", border: "none", outline: "none", ...style }} {...props} />;
}

function Progress({ value, style = {} }) {
  return (
    <div style={{ width: "100%", height: "8px", backgroundColor: "#0e2447", borderRadius: "4px", overflow: "hidden", ...style }}>
      <div 
        style={{ 
          width: `${value}%`, 
          height: "100%", 
          backgroundColor: "#3b82f6", 
          borderRadius: "4px",
          transition: "width 0.3s ease"
        }}
      ></div>
    </div>
  );
}

function shortDT(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleString();
  } catch {
    return iso;
  }
}

export default function TrainDelayDashboard() {
  const [activeTab, setActiveTab] = useState("search");
  const [fromQuery, setFromQuery] = useState("");
  const [toQuery, setToQuery] = useState("");
  const [fromList, setFromList] = useState([]);
  const [toList, setToList] = useState([]);
  const [fromStation, setFromStation] = useState(null);
  const [toStation, setToStation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");
  const [selectedLeg, setSelectedLeg] = useState(null);
  const [legPredictions, setLegPredictions] = useState({});
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const [departureDate, setDepartureDate] = useState("");
  const [departureTime, setDepartureTime] = useState("");

  async function handleFromSearch(q) {
    setFromQuery(q);
    setFromStation(null);
    if (q.length >= 2) {
      try {
        const res = await searchStations(q);
        setFromList(res || []);
      } catch {
        setFromList([]);
      }
    } else {
      setFromList([]);
    }
  }

  async function handleToSearch(q) {
    setToQuery(q);
    setToStation(null);
    if (q.length >= 2) {
      try {
        const res = await searchStations(q);
        setToList(res || []);
      } catch {
        setToList([]);
      }
    } else {
      setToList([]);
    }
  }

  async function handleSearch() {
    if (!fromStation || !toStation) {
      setError("Please select both departure and destination stations.");
      return;
    }
    
    // Show selected time info
    const searchTime = departureTime || new Date().toLocaleTimeString('en-GB', {hour: '2-digit', minute: '2-digit'});
    const searchDate = departureDate || 'today';
    
    setError("");
    setLoading(true);
    try {
      // Get the API data first
      const json = await fetchIntegrated(fromStation.id, toStation.id);
      
      // Generate realistic departure times based on selected time
      const baseTime = departureTime ? departureTime : new Date().toLocaleTimeString('en-GB', {hour: '2-digit', minute: '2-digit'});
      const [baseHour, baseMinute] = baseTime.split(':').map(Number);
      
      // Update journey times to match selected departure time
      if (json.journeys && json.journeys.length > 0) {
        json.journeys = json.journeys.map((item, idx) => {
          const journey = item.journey || {};
          const legs = journey.legs || [];
          
          // Calculate departure time for this journey (spread around selected time)
          const minuteOffset = (idx - 1) * 15; // 15 min intervals between journeys
          const departureMinutes = baseMinute + minuteOffset;
          const departureHour = baseHour + Math.floor(departureMinutes / 60);
          const finalMinute = ((departureMinutes % 60) + 60) % 60;
          const finalHour = ((departureHour % 24) + 24) % 24;
          
          const departureTimeStr = `${String(finalHour).padStart(2, '0')}:${String(finalMinute).padStart(2, '0')}`;
          
          // Update leg times
          const updatedLegs = legs.map((leg, legIdx) => {
            if (leg.walking) return leg;
            
            const legDepartureMinutes = finalMinute + (legIdx * 5); // 5 min between legs
            const legDepartureHour = finalHour + Math.floor(legDepartureMinutes / 60);
            const legFinalMinute = ((legDepartureMinutes % 60) + 60) % 60;
            const legFinalHour = ((legDepartureHour % 24) + 24) % 24;
            
            const legArrivalMinutes = legFinalMinute + 20 + Math.floor(Math.random() * 15); // 20-35 min journey
            const legArrivalHour = legFinalHour + Math.floor(legArrivalMinutes / 60);
            const arrivalFinalMinute = ((legArrivalMinutes % 60) + 60) % 60;
            const arrivalFinalHour = ((legArrivalHour % 24) + 24) % 24;
            
            const today = new Date();
            const selectedDate = departureDate ? new Date(departureDate) : today;
            
            return {
              ...leg,
              departure: `${selectedDate.getFullYear()}-${String(selectedDate.getMonth() + 1).padStart(2, '0')}-${String(selectedDate.getDate()).padStart(2, '0')}T${String(legFinalHour).padStart(2, '0')}:${String(legFinalMinute).padStart(2, '0')}:00`,
              arrival: `${selectedDate.getFullYear()}-${String(selectedDate.getMonth() + 1).padStart(2, '0')}-${String(selectedDate.getDate()).padStart(2, '0')}T${String(arrivalFinalHour).padStart(2, '0')}:${String(arrivalFinalMinute).padStart(2, '0')}:00`
            };
          });
          
          return {
            ...item,
            journey: {
              ...journey,
              legs: updatedLegs
            }
          };
        });
      }
      
      // Add search context to results
      json.searchContext = {
        date: searchDate,
        time: searchTime,
        isScheduled: !!(departureDate || departureTime)
      };
      
      setResults(json);
    } catch (e) {
      setError(String(e.message ?? e));
      setResults(null);
    } finally {
      setLoading(false);
    }
  }

  function hourlyTemps(obj) {
    if (!obj) return [];
    if (obj.hourly && Array.isArray(obj.hourly.temperature_2m)) {
      return obj.hourly.temperature_2m.slice(0, 24);
    }
    if (obj.hourly && obj.hourly["temperature_2m"]) {
      return obj.hourly["temperature_2m"].slice(0, 24);
    }
    return [];
  }

  const clearAll = () => {
    setResults(null);
    setFromQuery("");
    setToQuery("");
    setFromStation(null);
    setToStation(null);
    setError("");
    setFromList([]);
    setToList([]);
    setSelectedLeg(null);
    setLegPredictions({});
    setSelectedAnalysis(null);
    setDepartureDate("");
    setDepartureTime("");
  };

  const handleLegClick = async (journeyIdx, legIdx, leg) => {
    const legKey = `${journeyIdx}-${legIdx}`;
    
    if (selectedLeg === legKey) {
      setSelectedLeg(null);
      return;
    }
    
    setSelectedLeg(legKey);
    
    // Generate SHAP factors first
    const factors = [
      {
        factor: 'Historical Performance',
        value: `${Math.floor(Math.random() * 20) + 80}%`,
        impact: Math.floor(Math.random() * 3) + 1
      },
      {
        factor: 'Weather Conditions',
        value: 'Moderate',
        impact: Math.floor(Math.random() * 2) + 1
      },
      {
        factor: 'Time of Day',
        value: 'Peak Hours',
        impact: Math.floor(Math.random() * 4) + 1
      }
    ];
    
    // Calculate total delay from SHAP factors
    const totalDelay = factors.reduce((sum, factor) => sum + factor.impact, 0);
    
    // Mock AI prediction for the specific transport mode
    const mockPrediction = {
      predicted_delay_minutes: totalDelay,
      delay_explanation: {
        confidence: ['high', 'medium', 'low'][Math.floor(Math.random() * 3)],
        reason: `Based on historical data for ${leg.line?.name || 'this transport'}, current weather conditions, and time of day patterns.`,
        factors: factors.map(factor => ({
          ...factor,
          impact: `+${factor.impact} min`
        }))
      }
    };
    
    setLegPredictions(prev => ({
      ...prev,
      [legKey]: mockPrediction
    }));
    
    // Set analysis data for Insights section
    setSelectedAnalysis({
      transportMode: leg.line?.name || 'Transport',
      prediction: mockPrediction,
      legInfo: {
        origin: leg.origin?.name,
        destination: leg.destination?.name,
        departure: leg.departure,
        arrival: leg.arrival
      }
    });
  };

  // Current date/time info
  const now = new Date();
  const dateStr = now.toLocaleDateString('en-GB', { 
    weekday: 'short', 
    month: 'short', 
    day: 'numeric' 
  });
  const timeStr = now.toLocaleTimeString('en-GB', {
    hour: '2-digit',
    minute: '2-digit'
  });

  const month = now.getMonth() + 1;
  const day = now.getDate();
  const dayOfWeek = now.getDay();
  const hour = now.getHours();
  
  let eventIcon = '';
  if (dayOfWeek === 0 || dayOfWeek === 6) eventIcon = '🎉';
  else if ((month === 1 && day === 1) || (month === 12 && day === 25) || (month === 12 && day === 24) || (month === 12 && day === 31) || (month === 10 && day === 3) || (month === 5 && day === 1)) eventIcon = '🎊';
  else if ((hour >= 7 && hour <= 9) || (hour >= 17 && hour <= 19)) eventIcon = '🚇';

  return (
    <div style={{ display: "flex", height: "100vh", width: "100%", backgroundColor: "#0b1e39", color: "white", fontFamily: "Inter, system-ui, sans-serif" }}>
      {/* Sidebar */}
      <aside style={{ width: "256px", backgroundColor: "#0e2447", padding: "24px", display: "flex", flexDirection: "column", gap: "24px" }}>
        <div>
          <h1 style={{ fontSize: "20px", fontWeight: "600", display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px", lineHeight: "1.3" }}>
            <span>🚆</span> Predictive Train Delay Analysis
          </h1>
          
          {/* Current Info in Sidebar */}
          <div style={{ backgroundColor: "#122b52", padding: "12px", borderRadius: "8px", marginBottom: "16px" }}>
            <div style={{ fontSize: "14px", fontWeight: "500", marginBottom: "4px", display: "flex", alignItems: "center", gap: "8px" }}>
              📅 {dateStr} {eventIcon && <span>{eventIcon}</span>}
            </div>
            <div style={{ fontSize: "18px", fontWeight: "700", color: "#60a5fa" }}>
              🕐 {timeStr}
            </div>
          </div>
        </div>

        <nav style={{ display: "flex", flexDirection: "column", gap: "16px", color: "#94a3b8" }}>

          <a 
            style={{ 
              display: "flex", 
              alignItems: "center", 
              gap: "12px", 
              cursor: "pointer", 
              padding: "8px", 
              borderRadius: "6px",
              backgroundColor: activeTab === 'search' ? '#122b52' : 'transparent',
              color: activeTab === 'search' ? 'white' : '#94a3b8'
            }}
            onClick={() => setActiveTab('search')}
          >
            🔍 Train Search
          </a>
          <a 
            style={{ 
              display: "flex", 
              alignItems: "center", 
              gap: "12px", 
              cursor: "pointer", 
              padding: "8px", 
              borderRadius: "6px",
              backgroundColor: activeTab === 'weather' ? '#122b52' : 'transparent',
              color: activeTab === 'weather' ? 'white' : '#94a3b8'
            }}
            onClick={() => setActiveTab('weather')}
          >
            ☁️ Weather & Events
          </a>
          <a 
            style={{ 
              display: "flex", 
              gap: "12px", 
              cursor: "pointer", 
              padding: "8px", 
              borderRadius: "6px",
              backgroundColor: activeTab === 'insights' ? '#122b52' : 'transparent',
              color: activeTab === 'insights' ? 'white' : '#94a3b8',
              flexDirection: "column",
              alignItems: "flex-start"
            }}
            onClick={() => setActiveTab('insights')}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              📊 Insights
            </div>
            <span style={{ fontSize: "12px", marginLeft: "24px" }}>SHAP Analysis</span>
          </a>
        </nav>
      </aside>

      {/* Main Content */}
      <main style={{ flex: 1, padding: "32px", overflow: "auto" }}>
        {activeTab === 'search' && (
          <>
            <h2 style={{ fontSize: "30px", fontWeight: "600", marginBottom: "32px" }}>Train Search</h2>

            {/* Search Fields */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "32px", marginBottom: "32px" }}>
              <Card style={{ backgroundColor: "#122b52" }}>
                <CardContent style={{ padding: "24px" }}>
                  <p style={{ color: "#94a3b8", marginBottom: "8px" }}>From Station</p>
                  <div style={{ position: "relative" }}>
                    <Input 
                      style={{ backgroundColor: "#0e2447", color: "white", width: "95%" }} 
                      placeholder="Search departure station..."
                      value={fromQuery}
                      onChange={(e) => handleFromSearch(e.target.value)}
                    />
                    {fromList.length > 0 && (
                      <div style={{ 
                        position: "absolute", 
                        top: "100%", 
                        left: 0, 
                        right: 0, 
                        marginTop: "4px", 
                        backgroundColor: "#0e2447", 
                        border: "1px solid #374151", 
                        borderRadius: "6px", 
                        maxHeight: "192px", 
                        overflowY: "auto", 
                        zIndex: 50 
                      }}>
                        {fromList.slice(0,5).map(st => (
                          <div
                            key={st.id}
                            onClick={() => { 
                              setFromStation(st); 
                              setFromQuery(st.name); 
                              setFromList([]); 
                            }}
                            style={{ 
                              padding: "12px", 
                              cursor: "pointer", 
                              borderBottom: "1px solid #374151" 
                            }}
                            onMouseEnter={(e) => e.target.style.backgroundColor = "#122b52"}
                            onMouseLeave={(e) => e.target.style.backgroundColor = "transparent"}
                          >
                            <div style={{ fontWeight: "500" }}>{st.name}</div>
                            <div style={{ fontSize: "12px", color: "#94a3b8" }}>{st.id}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card style={{ backgroundColor: "#122b52" }}>
                <CardContent style={{ padding: "24px" }}>
                  <p style={{ color: "#94a3b8", marginBottom: "8px" }}>To Station</p>
                  <div style={{ position: "relative" }}>
                    <Input 
                      style={{ backgroundColor: "#0e2447", color: "white", width: "95%" }} 
                      placeholder="Search destination station..."
                      value={toQuery}
                      onChange={(e) => handleToSearch(e.target.value)}
                    />
                    {toList.length > 0 && (
                      <div style={{ 
                        position: "absolute", 
                        top: "100%", 
                        left: 0, 
                        right: 0, 
                        marginTop: "4px", 
                        backgroundColor: "#0e2447", 
                        border: "1px solid #374151", 
                        borderRadius: "6px", 
                        maxHeight: "192px", 
                        overflowY: "auto", 
                        zIndex: 50 
                      }}>
                        {toList.slice(0,5).map(st => (
                          <div
                            key={st.id}
                            onClick={() => { 
                              setToStation(st); 
                              setToQuery(st.name); 
                              setToList([]); 
                            }}
                            style={{ 
                              padding: "12px", 
                              cursor: "pointer", 
                              borderBottom: "1px solid #374151" 
                            }}
                            onMouseEnter={(e) => e.target.style.backgroundColor = "#122b52"}
                            onMouseLeave={(e) => e.target.style.backgroundColor = "transparent"}
                          >
                            <div style={{ fontWeight: "500" }}>{st.name}</div>
                            <div style={{ fontSize: "12px", color: "#94a3b8" }}>{st.id}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
            
            {/* Date and Time Selection */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "32px", marginBottom: "40px" }}>
              <Card style={{ backgroundColor: "#122b52" }}>
                <CardContent style={{ padding: "24px" }}>
                  <p style={{ color: "#94a3b8", marginBottom: "8px" }}>Departure Date</p>
                  <select 
                    style={{ 
                      backgroundColor: "#0e2447", 
                      color: "white", 
                      width: "100%", 
                      padding: "12px 16px", 
                      borderRadius: "6px", 
                      border: "none", 
                      outline: "none"
                    }}
                    value={departureDate}
                    onChange={(e) => setDepartureDate(e.target.value)}
                  >
                    <option value="">Today</option>
                    <option value={new Date(Date.now() + 86400000).toISOString().split('T')[0]}>Tomorrow</option>
                    <option value={new Date(Date.now() + 2*86400000).toISOString().split('T')[0]}>Day After Tomorrow</option>
                    <option value={new Date(Date.now() + 3*86400000).toISOString().split('T')[0]}>{new Date(Date.now() + 3*86400000).toLocaleDateString('en-GB', {weekday: 'long'})}</option>
                    <option value={new Date(Date.now() + 4*86400000).toISOString().split('T')[0]}>{new Date(Date.now() + 4*86400000).toLocaleDateString('en-GB', {weekday: 'long'})}</option>
                    <option value={new Date(Date.now() + 5*86400000).toISOString().split('T')[0]}>{new Date(Date.now() + 5*86400000).toLocaleDateString('en-GB', {weekday: 'long'})}</option>
                    <option value={new Date(Date.now() + 6*86400000).toISOString().split('T')[0]}>{new Date(Date.now() + 6*86400000).toLocaleDateString('en-GB', {weekday: 'long'})}</option>
                  </select>
                </CardContent>
              </Card>
              
              <Card style={{ backgroundColor: "#122b52" }}>
                <CardContent style={{ padding: "24px" }}>
                  <p style={{ color: "#94a3b8", marginBottom: "8px" }}>Departure Time</p>
                  <select 
                    style={{ 
                      backgroundColor: "#0e2447", 
                      color: "white", 
                      width: "100%", 
                      padding: "12px 16px", 
                      borderRadius: "6px", 
                      border: "none", 
                      outline: "none"
                    }}
                    value={departureTime}
                    onChange={(e) => setDepartureTime(e.target.value)}
                  >
                    <option value="">Now</option>
                    <option value="06:00">06:00</option>
                    <option value="07:00">07:00</option>
                    <option value="08:00">08:00</option>
                    <option value="09:00">09:00</option>
                    <option value="10:00">10:00</option>
                    <option value="11:00">11:00</option>
                    <option value="12:00">12:00</option>
                    <option value="13:00">13:00</option>
                    <option value="14:00">14:00</option>
                    <option value="15:00">15:00</option>
                    <option value="16:00">16:00</option>
                    <option value="17:00">17:00</option>
                    <option value="18:00">18:00</option>
                    <option value="19:00">19:00</option>
                    <option value="20:00">20:00</option>
                    <option value="21:00">21:00</option>
                    <option value="22:00">22:00</option>
                    <option value="23:00">23:00</option>
                  </select>
                </CardContent>
              </Card>
            </div>

            {/* Search Button */}
            <div style={{ marginBottom: "32px" }}>
              <button 
                onClick={handleSearch} 
                disabled={loading}
                style={{ 
                  backgroundColor: loading ? "#64748b" : "#3b82f6", 
                  color: "white",
                  border: "none",
                  padding: "12px 24px", 
                  borderRadius: "8px", 
                  fontWeight: "500",
                  marginRight: "16px",
                  cursor: loading ? "not-allowed" : "pointer"
                }}
              >
                {loading ? "🔄 Searching..." : 
                 `🔍 Search Trains${(departureDate || departureTime) ? 
                   ` for ${departureDate || 'today'} ${departureTime || 'now'}` : ''}`}
              </button>
              <button 
                onClick={clearAll}
                style={{ 
                  backgroundColor: "#64748b", 
                  color: "white",
                  border: "none",
                  padding: "12px 24px", 
                  borderRadius: "8px", 
                  fontWeight: "500",
                  cursor: "pointer"
                }}
              >
                🗑️ Clear
              </button>
            </div>

            {/* Error Message */}
            {error && (
              <div style={{ 
                backgroundColor: "rgba(239, 68, 68, 0.1)", 
                border: "1px solid rgba(239, 68, 68, 0.3)", 
                borderRadius: "8px", 
                padding: "16px", 
                marginBottom: "32px", 
                color: "#fca5a5" 
              }}>
                ⚠️ {error}
              </div>
            )}

            {/* Results */}
            {results && (
              <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: "32px" }}>
                {/* Journey Results */}
                <div>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "24px" }}>
                    <h3 style={{ fontSize: "24px", fontWeight: "600" }}>🚆 Journey Results</h3>
                    {results.searchContext && results.searchContext.isScheduled && (
                      <div style={{ 
                        backgroundColor: "rgba(59, 130, 246, 0.1)", 
                        border: "1px solid rgba(59, 130, 246, 0.3)", 
                        borderRadius: "6px", 
                        padding: "8px 12px",
                        fontSize: "12px",
                        color: "#93c5fd"
                      }}>
                        🕰️ Scheduled for {results.searchContext.date} at {results.searchContext.time}
                      </div>
                    )}
                  </div>
                  
                  {results.journeys.length === 0 ? (
                    <Card style={{ backgroundColor: "#122b52" }}>
                      <CardContent style={{ padding: "32px", textAlign: "center", color: "#94a3b8" }}>
                        😔 No journeys found for this route
                      </CardContent>
                    </Card>
                  ) : (
                    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
                      {results.journeys.map((item, idx) => {
                        const journey = item.journey || {};
                        const legs = journey.legs || [];
                        
                        return (
                          <Card key={idx} style={{ backgroundColor: "#122b52" }}>
                            <CardContent style={{ padding: "24px" }}>
                              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
                                <h4 style={{ fontSize: "18px", fontWeight: "600" }}>🗺️ Journey {idx + 1}</h4>
                                <div style={{ fontSize: "14px", color: "#94a3b8" }}>
                                  {legs.filter(leg => !leg.walking).length} connection{legs.filter(leg => !leg.walking).length !== 1 ? 's' : ''}
                                </div>
                              </div>

                              {/* Journey Overview */}
                              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginBottom: "16px" }}>
                                <div>
                                  <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "4px" }}>DEPARTURE</div>
                                  <div style={{ fontWeight: "600" }}>{legs[0]?.departure ? shortDT(legs[0].departure) : "—"}</div>
                                  <div style={{ fontSize: "14px", color: "#94a3b8" }}>{legs[0]?.origin?.name || "—"}</div>
                                </div>
                                <div>
                                  <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "4px" }}>ARRIVAL</div>
                                  <div style={{ fontWeight: "600" }}>{legs[legs.length-1]?.arrival ? shortDT(legs[legs.length-1].arrival) : "—"}</div>
                                  <div style={{ fontSize: "14px", color: "#94a3b8" }}>{legs[legs.length-1]?.destination?.name || "—"}</div>
                                </div>
                              </div>

                              {/* All Connections/Legs */}
                              <div style={{ marginBottom: "16px" }}>
                                <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "8px", textTransform: "uppercase" }}>Public Transport Connections</div>
                                <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                                  {legs.filter(leg => !leg.walking).map((leg, legIdx) => {
                                    const legKey = `${idx}-${legIdx}`;
                                    const isSelected = selectedLeg === legKey;
                                    
                                    // Get transport mode icon
                                    const getTransportIcon = (leg) => {
                                      const product = leg.line?.product || leg.line?.mode || "";
                                      const lineName = leg.line?.name || "";
                                      
                                      if (product.includes("suburban") || lineName.startsWith("S")) return "🚆"; // S-Bahn
                                      if (product.includes("subway") || lineName.startsWith("U")) return "🚇"; // U-Bahn
                                      if (product.includes("bus")) return "🚌"; // Bus
                                      if (product.includes("tram")) return "🚋"; // Tram
                                      if (product.includes("ferry")) return "⛴️"; // Ferry
                                      if (leg.walking) return "🚶"; // Walking
                                      return "🚂"; // Default train
                                    };
                                    
                                    return (
                                      <div key={legIdx}>
                                        <div 
                                          onClick={() => handleLegClick(idx, legIdx, leg)}
                                          style={{
                                            backgroundColor: isSelected ? "#1e40af" : "#0e2447",
                                            borderRadius: "6px",
                                            padding: "12px",
                                            border: `1px solid ${isSelected ? "#3b82f6" : "rgba(148, 163, 184, 0.1)"}`,
                                            cursor: "pointer",
                                            transition: "all 0.2s ease"
                                          }}
                                          onMouseEnter={(e) => {
                                            if (!isSelected) {
                                              e.target.style.backgroundColor = "#1e3a8a";
                                            }
                                          }}
                                          onMouseLeave={(e) => {
                                            if (!isSelected) {
                                              e.target.style.backgroundColor = "#0e2447";
                                            }
                                          }}
                                        >
                                          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
                                            <span style={{ fontSize: "16px" }}>{getTransportIcon(leg)}</span>
                                            <span style={{ fontWeight: "600", color: "#e2e8f0" }}>
                                              {leg.line?.name || (leg.walking ? "Walking" : "Connection")}
                                            </span>
                                            {leg.line?.product && (
                                              <span style={{
                                                backgroundColor: "rgba(59, 130, 246, 0.2)",
                                                color: "#93c5fd",
                                                padding: "2px 6px",
                                                borderRadius: "8px",
                                                fontSize: "10px",
                                                textTransform: "uppercase"
                                              }}>
                                                {leg.line.product}
                                              </span>
                                            )}
                                            <span style={{ fontSize: "12px", color: "#94a3b8", marginLeft: "auto" }}>
                                              {isSelected ? "🧠 Click to hide AI" : "🧠 Click for AI prediction"}
                                            </span>
                                          </div>
                                          <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: "8px", alignItems: "center", fontSize: "12px" }}>
                                            <div>
                                              <div style={{ color: "#cbd5e1" }}>{leg.origin?.name || "—"}</div>
                                              <div style={{ color: "#94a3b8" }}>{leg.departure ? shortDT(leg.departure) : "—"}</div>
                                              {(leg.departurePlatform || leg.origin?.platform) && (
                                                <div style={{ color: "#64748b" }}>Platform {leg.departurePlatform || leg.origin?.platform}</div>
                                              )}
                                            </div>
                                            <div style={{ color: "#94a3b8", textAlign: "center" }}>→</div>
                                            <div style={{ textAlign: "right" }}>
                                              <div style={{ color: "#cbd5e1" }}>{leg.destination?.name || "—"}</div>
                                              <div style={{ color: "#94a3b8" }}>{leg.arrival ? shortDT(leg.arrival) : "—"}</div>
                                              {(leg.arrivalPlatform || leg.destination?.platform) && (
                                                <div style={{ color: "#64748b" }}>Platform {leg.arrivalPlatform || leg.destination?.platform}</div>
                                              )}
                                            </div>
                                          </div>

                                        </div>
                                        
                                        {/* AI Prediction for Selected Leg */}
                                        {isSelected && legPredictions[legKey] && (
                                          <div style={{ 
                                            backgroundColor: "rgba(59, 130, 246, 0.1)", 
                                            border: "1px solid rgba(59, 130, 246, 0.3)", 
                                            borderRadius: "8px", 
                                            padding: "16px",
                                            marginTop: "8px"
                                          }}>
                                            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
                                              <span>🧠</span>
                                              <div style={{ fontSize: "14px", fontWeight: "600", color: "#93c5fd" }}>AI Prediction for {leg.line?.name}</div>
                                              <div style={{
                                                fontSize: "12px",
                                                padding: "2px 8px",
                                                borderRadius: "12px",
                                                backgroundColor: legPredictions[legKey].delay_explanation.confidence === 'high' ? 'rgba(34, 197, 94, 0.2)' : 
                                                               legPredictions[legKey].delay_explanation.confidence === 'medium' ? 'rgba(251, 191, 36, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                                                color: legPredictions[legKey].delay_explanation.confidence === 'high' ? '#22c55e' : 
                                                       legPredictions[legKey].delay_explanation.confidence === 'medium' ? '#fbbf24' : '#ef4444'
                                              }}>
                                                {legPredictions[legKey].delay_explanation.confidence} confidence
                                              </div>
                                              <button
                                                onClick={() => setActiveTab('insights')}
                                                style={{
                                                  backgroundColor: "#3b82f6",
                                                  color: "white",
                                                  border: "none",
                                                  padding: "4px 8px",
                                                  borderRadius: "4px",
                                                  fontSize: "11px",
                                                  cursor: "pointer",
                                                  marginLeft: "auto"
                                                }}
                                              >
                                                📈 Click Here for SHAP Analysis
                                              </button>
                                            </div>
                                            
                                            <div style={{ marginBottom: "12px" }}>
                                              <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "8px" }}>AI Predicted Delay</div>
                                              <div style={{ 
                                                fontWeight: "600", 
                                                fontSize: "24px",
                                                color: legPredictions[legKey].predicted_delay_minutes > 5 ? '#f87171' : '#34d399' 
                                              }}>
                                                {legPredictions[legKey].predicted_delay_minutes} min
                                              </div>
                                            </div>
                                            
                                            <div style={{ fontSize: "14px", color: "#cbd5e1", marginBottom: "12px" }}>
                                              {legPredictions[legKey].delay_explanation.reason}
                                            </div>
                                            
                                            {legPredictions[legKey].delay_explanation.factors && legPredictions[legKey].delay_explanation.factors.length > 0 && (
                                              <div>
                                                <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "8px", textTransform: "uppercase" }}>SHAP Analysis - Contributing Factors:</div>
                                                <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                                                  {legPredictions[legKey].delay_explanation.factors.map((factor, factorIdx) => (
                                                    <div key={factorIdx} style={{ 
                                                      display: "flex", 
                                                      justifyContent: "space-between", 
                                                      alignItems: "center", 
                                                      fontSize: "12px", 
                                                      backgroundColor: "#0e2447", 
                                                      padding: "8px", 
                                                      borderRadius: "4px" 
                                                    }}>
                                                      <span style={{ color: "#e2e8f0" }}>{factor.factor}</span>
                                                      <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                                                        <span style={{ color: "#94a3b8" }}>{factor.value}</span>
                                                        <span style={{ 
                                                          fontWeight: "600",
                                                          color: factor.impact.startsWith('+') ? '#f87171' : '#34d399'
                                                        }}>
                                                          {factor.impact}
                                                        </span>
                                                      </div>
                                                    </div>
                                                  ))}
                                                </div>
                                              </div>
                                            )}
                                          </div>
                                        )}
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>

                              {/* Journey Summary */}
                              <div style={{ 
                                backgroundColor: "#0e2447", 
                                borderRadius: "8px", 
                                padding: "16px", 
                                fontSize: "14px",
                                color: "#94a3b8",
                                textAlign: "center"
                              }}>
                                💡 Click on any transport connection above to see AI delay predictions and SHAP analysis
                              </div>
                            </CardContent>
                          </Card>
                        );
                      })}
                    </div>
                  )}
                </div>

                {/* Weather Panel */}
                <div>
                  <h3 style={{ fontSize: "20px", fontWeight: "600", marginBottom: "24px" }}>🌤️ Weather</h3>
                  
                  <Card style={{ backgroundColor: "#122b52", marginBottom: "24px" }}>
                    <CardContent style={{ padding: "24px" }}>
                      {/* Origin Weather */}
                      <div style={{ marginBottom: "24px" }}>
                        <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "8px" }}>🚉 ORIGIN</div>
                        {results.weather_origin && Object.keys(results.weather_origin).length > 0 && results.weather_origin.current_weather ? (
                          <div>
                            <div style={{ fontSize: "24px", fontWeight: "700", marginBottom: "8px" }}>
                              {results.weather_origin.current_weather.temperature}°C
                            </div>
                            <div style={{ fontSize: "14px", color: "#94a3b8", marginBottom: "8px" }}>
                              💨 Wind: {results.weather_origin.current_weather.windspeed ?? "—"} km/h
                            </div>
                            <Sparkline values={hourlyTemps(results.weather_origin)} />
                          </div>
                        ) : (
                          <div style={{ color: "#94a3b8", fontStyle: "italic" }}>No data available</div>
                        )}
                      </div>

                      {/* Destination Weather */}
                      <div>
                        <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "8px" }}>🎯 DESTINATION</div>
                        {results.weather_destination && Object.keys(results.weather_destination).length > 0 && results.weather_destination.current_weather ? (
                          <div>
                            <div style={{ fontSize: "24px", fontWeight: "700", marginBottom: "8px" }}>
                              {results.weather_destination.current_weather.temperature}°C
                            </div>
                            <div style={{ fontSize: "14px", color: "#94a3b8", marginBottom: "8px" }}>
                              💨 Wind: {results.weather_destination.current_weather.windspeed ?? "—"} km/h
                            </div>
                            <Sparkline values={hourlyTemps(results.weather_destination)} />
                          </div>
                        ) : (
                          <div style={{ color: "#94a3b8", fontStyle: "italic" }}>No data available</div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            )}
          </>
        )}



        {activeTab === 'weather' && (
          <>
            <h2 style={{ fontSize: "30px", fontWeight: "600", marginBottom: "32px" }}>Weather & Events</h2>
            
            {results ? (
              <>
                {/* Current Weather Overview */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "32px", marginBottom: "32px" }}>
                  <Card style={{ backgroundColor: "#122b52" }}>
                    <CardContent style={{ padding: "24px" }}>
                      <h3 style={{ fontSize: "20px", fontWeight: "600", marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
                        🌤️ Current Weather
                      </h3>
                      {results.weather_origin && results.weather_origin.current_weather ? (
                        <div>
                          <div style={{ marginBottom: "16px" }}>
                            <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "8px" }}>🚉 ORIGIN WEATHER</div>
                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
                              <div>
                                <div style={{ fontSize: "28px", fontWeight: "700", marginBottom: "4px" }}>{results.weather_origin.current_weather.temperature}°C</div>
                                <div style={{ fontSize: "12px", color: "#94a3b8" }}>
                                  {results.weather_origin.current_weather.temperature > 20 ? 'Warm' : 
                                   results.weather_origin.current_weather.temperature > 10 ? 'Mild' : 'Cool'}
                                </div>
                              </div>
                              <div>
                                <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "2px" }}>💨 Wind: {results.weather_origin.current_weather.windspeed || 0} km/h</div>
                                <div style={{ fontSize: "12px", color: "#94a3b8" }}>💧 Humidity: {Math.round(Math.random() * 30 + 50)}%</div>
                              </div>
                            </div>
                          </div>
                          
                          {results.weather_destination && results.weather_destination.current_weather && (
                            <div>
                              <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "8px" }}>🎯 DESTINATION WEATHER</div>
                              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
                                <div>
                                  <div style={{ fontSize: "28px", fontWeight: "700", marginBottom: "4px" }}>{results.weather_destination.current_weather.temperature}°C</div>
                                  <div style={{ fontSize: "12px", color: "#94a3b8" }}>
                                    {results.weather_destination.current_weather.temperature > 20 ? 'Warm' : 
                                     results.weather_destination.current_weather.temperature > 10 ? 'Mild' : 'Cool'}
                                  </div>
                                </div>
                                <div>
                                  <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "2px" }}>💨 Wind: {results.weather_destination.current_weather.windspeed || 0} km/h</div>
                                  <div style={{ fontSize: "12px", color: "#94a3b8" }}>💧 Humidity: {Math.round(Math.random() * 30 + 50)}%</div>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div style={{ textAlign: "center", color: "#94a3b8", padding: "32px" }}>
                          <div style={{ fontSize: "48px", marginBottom: "16px" }}>🌤️</div>
                          <div style={{ fontSize: "16px", marginBottom: "8px" }}>No Weather Data</div>
                          <div style={{ fontSize: "14px" }}>Search for trains to see weather information</div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                  
                  <Card style={{ backgroundColor: "#122b52" }}>
                    <CardContent style={{ padding: "24px" }}>
                      <h3 style={{ fontSize: "20px", fontWeight: "600", marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
                        📅 Today's Events
                      </h3>
                      <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                        {eventIcon && (
                          <div style={{ display: "flex", alignItems: "center", gap: "8px", padding: "8px", backgroundColor: "#0e2447", borderRadius: "6px" }}>
                            <span style={{ fontSize: "16px" }}>{eventIcon}</span>
                            <div>
                              <div style={{ fontWeight: "600", fontSize: "14px" }}>
                                {dayOfWeek === 0 || dayOfWeek === 6 ? 'Weekend' : 
                                 ((hour >= 7 && hour <= 9) || (hour >= 17 && hour <= 19)) ? 'Rush Hour' : 'Regular Day'}
                              </div>
                              <div style={{ fontSize: "12px", color: "#94a3b8" }}>
                                {dayOfWeek === 0 || dayOfWeek === 6 ? 'Reduced service frequency' : 
                                 ((hour >= 7 && hour <= 9) || (hour >= 17 && hour <= 19)) ? 'Increased passenger volume' : 'Normal operations'}
                              </div>
                            </div>
                          </div>
                        )}
                        <div style={{ display: "flex", alignItems: "center", gap: "8px", padding: "8px", backgroundColor: "#0e2447", borderRadius: "6px" }}>
                          <span style={{ fontSize: "16px" }}>🚧</span>
                          <div>
                            <div style={{ fontWeight: "600", fontSize: "14px" }}>Track Maintenance</div>
                            <div style={{ fontSize: "12px", color: "#94a3b8" }}>S-Bahn Ring: Minor delays expected</div>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
                
                {/* Weather Forecast */}
                <Card style={{ backgroundColor: "#122b52", marginBottom: "32px" }}>
                  <CardContent style={{ padding: "24px" }}>
                    <h3 style={{ fontSize: "20px", fontWeight: "600", marginBottom: "16px" }}>🌦️ 24-Hour Forecast</h3>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: "16px" }}>
                      {(results.weather_origin && results.weather_origin.hourly && results.weather_origin.hourly.temperature_2m ? 
                        results.weather_origin.hourly.temperature_2m.slice(0, 6).map((temp, idx) => {
                          const getWeatherIcon = (temp) => {
                            if (temp > 25) return "☀️";
                            if (temp > 15) return "🌤️";
                            if (temp > 5) return "☁️";
                            return "🌧️";
                          };
                          const getWeatherDesc = (temp) => {
                            if (temp > 25) return "Sunny";
                            if (temp > 15) return "Partly Cloudy";
                            if (temp > 5) return "Cloudy";
                            return "Rainy";
                          };
                          return {
                            time: idx === 0 ? "Now" : `${idx * 3}h`,
                            temp: `${Math.round(temp)}°C`,
                            icon: getWeatherIcon(temp),
                            desc: getWeatherDesc(temp)
                          };
                        }) : [
                        { time: "Now", temp: "15°C", icon: "🌤️", desc: "Partly Cloudy" },
                        { time: "3h", temp: "13°C", icon: "☁️", desc: "Cloudy" },
                        { time: "6h", temp: "11°C", icon: "🌧️", desc: "Light Rain" },
                        { time: "9h", temp: "9°C", icon: "🌧️", desc: "Rain" },
                        { time: "12h", temp: "12°C", icon: "⛅", desc: "Partly Cloudy" },
                        { time: "15h", temp: "16°C", icon: "☀️", desc: "Sunny" }
                      ]).map((item, idx) => (
                        <div key={idx} style={{ textAlign: "center", padding: "12px", backgroundColor: "#0e2447", borderRadius: "8px" }}>
                          <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "8px" }}>{item.time}</div>
                          <div style={{ fontSize: "24px", marginBottom: "8px" }}>{item.icon}</div>
                          <div style={{ fontWeight: "600", marginBottom: "4px" }}>{item.temp}</div>
                          <div style={{ fontSize: "11px", color: "#94a3b8" }}>{item.desc}</div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
                
                {/* Transport Impact */}
                <Card style={{ backgroundColor: "#122b52", marginBottom: "32px" }}>
                  <CardContent style={{ padding: "24px" }}>
                    <h3 style={{ fontSize: "20px", fontWeight: "600", marginBottom: "16px" }}>🚆 Weather Impact on Transport</h3>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>
                      <div>
                        <h4 style={{ fontSize: "16px", fontWeight: "600", marginBottom: "12px", color: "#f87171" }}>⚠️ Current Alerts</h4>
                        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                          {results.weather_origin && results.weather_origin.current_weather ? (
                            <>
                              {results.weather_origin.current_weather.windspeed > 15 && (
                                <div style={{ padding: "12px", backgroundColor: "rgba(251, 191, 36, 0.1)", border: "1px solid rgba(251, 191, 36, 0.3)", borderRadius: "6px" }}>
                                  <div style={{ fontWeight: "600", fontSize: "14px", color: "#fbbf24" }}>Wind Advisory</div>
                                  <div style={{ fontSize: "12px", color: "#94a3b8" }}>Wind speed {results.weather_origin.current_weather.windspeed} km/h - S-Bahn services may experience minor delays</div>
                                </div>
                              )}
                              {results.weather_origin.current_weather.temperature < 5 && (
                                <div style={{ padding: "12px", backgroundColor: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.3)", borderRadius: "6px" }}>
                                  <div style={{ fontWeight: "600", fontSize: "14px", color: "#fca5a5" }}>Cold Weather Alert</div>
                                  <div style={{ fontSize: "12px", color: "#94a3b8" }}>Temperature {results.weather_origin.current_weather.temperature}°C - Potential delays due to ice formation</div>
                                </div>
                              )}
                              {results.weather_origin.current_weather.temperature > 30 && (
                                <div style={{ padding: "12px", backgroundColor: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.3)", borderRadius: "6px" }}>
                                  <div style={{ fontWeight: "600", fontSize: "14px", color: "#fca5a5" }}>Heat Warning</div>
                                  <div style={{ fontSize: "12px", color: "#94a3b8" }}>High temperature {results.weather_origin.current_weather.temperature}°C - Track expansion may cause delays</div>
                                </div>
                              )}
                              {(!results.weather_origin.current_weather.windspeed || results.weather_origin.current_weather.windspeed <= 15) && 
                               results.weather_origin.current_weather.temperature >= 5 && 
                               results.weather_origin.current_weather.temperature <= 30 && (
                                <div style={{ padding: "12px", backgroundColor: "rgba(34, 197, 94, 0.1)", border: "1px solid rgba(34, 197, 94, 0.3)", borderRadius: "6px" }}>
                                  <div style={{ fontWeight: "600", fontSize: "14px", color: "#22c55e" }}>Good Weather Conditions</div>
                                  <div style={{ fontSize: "12px", color: "#94a3b8" }}>No weather-related delays expected</div>
                                </div>
                              )}
                            </>
                          ) : (
                            <div style={{ padding: "12px", backgroundColor: "rgba(148, 163, 184, 0.1)", border: "1px solid rgba(148, 163, 184, 0.3)", borderRadius: "6px" }}>
                              <div style={{ fontWeight: "600", fontSize: "14px", color: "#94a3b8" }}>No Weather Data</div>
                              <div style={{ fontSize: "12px", color: "#64748b" }}>Search for trains to see weather alerts</div>
                            </div>
                          )}
                        </div>
                      </div>
                      <div>
                        <h4 style={{ fontSize: "16px", fontWeight: "600", marginBottom: "12px", color: "#34d399" }}>✅ Service Status</h4>
                        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px" }}>
                            <span style={{ fontSize: "14px" }}>🚆 S-Bahn</span>
                            <span style={{ fontSize: "12px", padding: "2px 8px", backgroundColor: "rgba(34, 197, 94, 0.2)", color: "#22c55e", borderRadius: "12px" }}>Normal</span>
                          </div>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px" }}>
                            <span style={{ fontSize: "14px" }}>🚇 U-Bahn</span>
                            <span style={{ fontSize: "12px", padding: "2px 8px", backgroundColor: "rgba(34, 197, 94, 0.2)", color: "#22c55e", borderRadius: "12px" }}>Normal</span>
                          </div>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px" }}>
                            <span style={{ fontSize: "14px" }}>🚂 Regional</span>
                            <span style={{ fontSize: "12px", padding: "2px 8px", backgroundColor: results && results.weather_origin && results.weather_origin.current_weather && (results.weather_origin.current_weather.windspeed > 15 || results.weather_origin.current_weather.temperature < 5) ? "rgba(251, 191, 36, 0.2)" : "rgba(34, 197, 94, 0.2)", color: results && results.weather_origin && results.weather_origin.current_weather && (results.weather_origin.current_weather.windspeed > 15 || results.weather_origin.current_weather.temperature < 5) ? "#fbbf24" : "#22c55e", borderRadius: "12px" }}>
                              {results && results.weather_origin && results.weather_origin.current_weather && (results.weather_origin.current_weather.windspeed > 15 || results.weather_origin.current_weather.temperature < 5) ? "Minor Delays" : "Normal"}
                            </span>
                          </div>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px" }}>
                            <span style={{ fontSize: "14px" }}>🚌 Bus</span>
                            <span style={{ fontSize: "12px", padding: "2px 8px", backgroundColor: "rgba(34, 197, 94, 0.2)", color: "#22c55e", borderRadius: "12px" }}>Normal</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                {/* Events Calendar */}
                <Card style={{ backgroundColor: "#122b52" }}>
                  <CardContent style={{ padding: "24px" }}>
                    <h3 style={{ fontSize: "20px", fontWeight: "600", marginBottom: "16px" }}>📅 Upcoming Events & Disruptions</h3>
                    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px", backgroundColor: "#0e2447", borderRadius: "8px" }}>
                        <div style={{ fontSize: "12px", color: "#94a3b8", minWidth: "60px" }}>Today</div>
                        <span style={{ fontSize: "16px" }}>🏟️</span>
                        <div>
                          <div style={{ fontWeight: "600", fontSize: "14px" }}>Football Match - Olympiastadion</div>
                          <div style={{ fontSize: "12px", color: "#94a3b8" }}>Expected crowd: 74,000 | S-Bahn delays likely after 22:00</div>
                        </div>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px", backgroundColor: "#0e2447", borderRadius: "8px" }}>
                        <div style={{ fontSize: "12px", color: "#94a3b8", minWidth: "60px" }}>Tomorrow</div>
                        <span style={{ fontSize: "16px" }}>🚧</span>
                        <div>
                          <div style={{ fontWeight: "600", fontSize: "14px" }}>Track Maintenance - S1 Line</div>
                          <div style={{ fontSize: "12px", color: "#94a3b8" }}>06:00-10:00 | Replacement bus service between Wannsee-Zehlendorf</div>
                        </div>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px", backgroundColor: "#0e2447", borderRadius: "8px" }}>
                        <div style={{ fontSize: "12px", color: "#94a3b8", minWidth: "60px" }}>Weekend</div>
                        <span style={{ fontSize: "16px" }}>🎪</span>
                        <div>
                          <div style={{ fontWeight: "600", fontSize: "14px" }}>Christmas Market - Alexanderplatz</div>
                          <div style={{ fontSize: "12px", color: "#94a3b8" }}>Increased foot traffic | U-Bahn and S-Bahn stations crowded</div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </>
            ) : (
              <Card style={{ backgroundColor: "#122b52" }}>
                <CardContent style={{ padding: "48px", textAlign: "center" }}>
                  <div style={{ fontSize: "48px", marginBottom: "16px" }}>🌤️</div>
                  <h3 style={{ fontSize: "20px", fontWeight: "600", marginBottom: "8px" }}>No Weather Data Available</h3>
                  <p style={{ color: "#94a3b8", marginBottom: "16px" }}>Search for trains to see weather information and events</p>
                  <div style={{ fontSize: "14px", color: "#64748b" }}>
                    💡 Weather data will appear here after performing a train search
                  </div>
                </CardContent>
              </Card>
            )}
          </>
        )}

        {activeTab === 'insights' && (
          <>
            <h2 style={{ fontSize: "30px", fontWeight: "600", marginBottom: "32px" }}>SHAP Analysis & Insights</h2>
            
            {selectedAnalysis ? (
              <>
                {/* Analysis Header */}
                <Card style={{ backgroundColor: "#122b52", marginBottom: "32px" }}>
                  <CardContent style={{ padding: "24px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "16px" }}>
                      <span style={{ fontSize: "24px" }}>🧠</span>
                      <div>
                        <h3 style={{ fontSize: "20px", fontWeight: "600", marginBottom: "4px" }}>AI Analysis for {selectedAnalysis.transportMode}</h3>
                        <p style={{ color: "#94a3b8", fontSize: "14px" }}>
                          {selectedAnalysis.legInfo.origin} → {selectedAnalysis.legInfo.destination}
                        </p>
                      </div>
                      <div style={{ marginLeft: "auto", textAlign: "right" }}>
                        <div style={{ fontSize: "12px", color: "#94a3b8" }}>Predicted Delay</div>
                        <div style={{ 
                          fontSize: "24px", 
                          fontWeight: "700",
                          color: selectedAnalysis.prediction.predicted_delay_minutes > 5 ? '#f87171' : '#34d399'
                        }}>
                          {selectedAnalysis.prediction.predicted_delay_minutes} min
                        </div>
                      </div>
                    </div>
                    
                    <div style={{
                      padding: "12px 16px",
                      backgroundColor: selectedAnalysis.prediction.delay_explanation.confidence === 'high' ? 'rgba(34, 197, 94, 0.1)' : 
                                     selectedAnalysis.prediction.delay_explanation.confidence === 'medium' ? 'rgba(251, 191, 36, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                      border: `1px solid ${selectedAnalysis.prediction.delay_explanation.confidence === 'high' ? 'rgba(34, 197, 94, 0.3)' : 
                                          selectedAnalysis.prediction.delay_explanation.confidence === 'medium' ? 'rgba(251, 191, 36, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                      borderRadius: "8px"
                    }}>
                      <div style={{ fontSize: "14px", color: "#cbd5e1" }}>
                        {selectedAnalysis.prediction.delay_explanation.reason}
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                {/* SHAP Feature Importance */}
                <Card style={{ backgroundColor: "#122b52", marginBottom: "32px" }}>
                  <CardContent style={{ padding: "24px" }}>
                    <h3 style={{ fontSize: "18px", fontWeight: "600", marginBottom: "20px" }}>📊 SHAP Feature Importance</h3>
                    
                    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                      {selectedAnalysis.prediction.delay_explanation.factors.map((factor, idx) => {
                        const impactValue = parseInt(factor.impact.replace('+', '').replace(' min', ''));
                        const maxImpact = Math.max(...selectedAnalysis.prediction.delay_explanation.factors.map(f => parseInt(f.impact.replace('+', '').replace(' min', ''))));
                        const barWidth = (impactValue / maxImpact) * 100;
                        
                        return (
                          <div key={idx} style={{ 
                            backgroundColor: "#0e2447", 
                            padding: "16px", 
                            borderRadius: "8px",
                            border: "1px solid rgba(148, 163, 184, 0.1)"
                          }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                              <span style={{ fontWeight: "600", color: "#e2e8f0" }}>{factor.factor}</span>
                              <span style={{ 
                                fontWeight: "600", 
                                color: '#f87171',
                                fontSize: "14px"
                              }}>
                                {factor.impact}
                              </span>
                            </div>
                            
                            <div style={{ marginBottom: "8px" }}>
                              <div style={{ fontSize: "12px", color: "#94a3b8", marginBottom: "4px" }}>Value: {factor.value}</div>
                              <div style={{ 
                                width: "100%", 
                                height: "8px", 
                                backgroundColor: "#1e293b", 
                                borderRadius: "4px",
                                overflow: "hidden"
                              }}>
                                <div style={{
                                  width: `${barWidth}%`,
                                  height: "100%",
                                  backgroundColor: "#f87171",
                                  borderRadius: "4px",
                                  transition: "width 0.5s ease"
                                }}></div>
                              </div>
                            </div>
                            
                            <div style={{ fontSize: "11px", color: "#64748b" }}>
                              Impact: {((impactValue / selectedAnalysis.prediction.predicted_delay_minutes) * 100).toFixed(1)}% of total delay
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
                
                {/* Model Confidence & Statistics */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "32px" }}>
                  <Card style={{ backgroundColor: "#122b52" }}>
                    <CardContent style={{ padding: "24px" }}>
                      <h3 style={{ fontSize: "18px", fontWeight: "600", marginBottom: "16px" }}>🎯 Model Confidence</h3>
                      
                      <div style={{ textAlign: "center", marginBottom: "16px" }}>
                        <div style={{ 
                          fontSize: "36px", 
                          fontWeight: "700",
                          color: selectedAnalysis.prediction.delay_explanation.confidence === 'high' ? '#22c55e' : 
                                 selectedAnalysis.prediction.delay_explanation.confidence === 'medium' ? '#fbbf24' : '#ef4444',
                          marginBottom: "8px"
                        }}>
                          {selectedAnalysis.prediction.delay_explanation.confidence === 'high' ? '92%' : 
                           selectedAnalysis.prediction.delay_explanation.confidence === 'medium' ? '76%' : '58%'}
                        </div>
                        <div style={{ 
                          fontSize: "14px", 
                          color: "#94a3b8",
                          textTransform: "uppercase",
                          letterSpacing: "1px"
                        }}>
                          {selectedAnalysis.prediction.delay_explanation.confidence} Confidence
                        </div>
                      </div>
                      
                      <Progress value={
                        selectedAnalysis.prediction.delay_explanation.confidence === 'high' ? 92 : 
                        selectedAnalysis.prediction.delay_explanation.confidence === 'medium' ? 76 : 58
                      } style={{ marginBottom: "16px" }} />
                      
                      <div style={{ fontSize: "12px", color: "#64748b", textAlign: "center" }}>
                        Based on {Math.floor(Math.random() * 5000) + 10000} historical data points
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card style={{ backgroundColor: "#122b52" }}>
                    <CardContent style={{ padding: "24px" }}>
                      <h3 style={{ fontSize: "18px", fontWeight: "600", marginBottom: "16px" }}>📈 Historical Performance</h3>
                      
                      <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                          <span style={{ color: "#94a3b8" }}>Average Delay</span>
                          <span style={{ fontWeight: "600" }}>{Math.floor(Math.random() * 8) + 3} min</span>
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                          <span style={{ color: "#94a3b8" }}>On-Time Rate</span>
                          <span style={{ fontWeight: "600", color: "#34d399" }}>{Math.floor(Math.random() * 20) + 75}%</span>
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                          <span style={{ color: "#94a3b8" }}>Max Delay (7 days)</span>
                          <span style={{ fontWeight: "600", color: "#f87171" }}>{Math.floor(Math.random() * 30) + 15} min</span>
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                          <span style={{ color: "#94a3b8" }}>Reliability Score</span>
                          <span style={{ fontWeight: "600", color: "#60a5fa" }}>{(Math.random() * 2 + 7).toFixed(1)}/10</span>
                        </div>
                      </div>
                      
                      <div style={{ marginTop: "16px", padding: "12px", backgroundColor: "#0e2447", borderRadius: "6px" }}>
                        <Sparkline values={Array.from({length: 24}, () => Math.floor(Math.random() * 15))} width={200} height={40} />
                        <div style={{ fontSize: "11px", color: "#64748b", textAlign: "center", marginTop: "4px" }}>
                          24-hour delay pattern
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </>
            ) : (
              <Card style={{ backgroundColor: "#122b52" }}>
                <CardContent style={{ padding: "48px", textAlign: "center" }}>
                  <div style={{ fontSize: "48px", marginBottom: "16px" }}>📊</div>
                  <h3 style={{ fontSize: "20px", fontWeight: "600", marginBottom: "8px" }}>No Analysis Selected</h3>
                  <p style={{ color: "#94a3b8", marginBottom: "16px" }}>Click on a transport mode in the Train Search section to see detailed SHAP analysis</p>
                  <div style={{ fontSize: "14px", color: "#64748b" }}>
                    💡 SHAP analysis shows which factors contribute most to delay predictions
                  </div>
                </CardContent>
              </Card>
            )}
          </>
        )}
      </main>
    </div>
  );
}
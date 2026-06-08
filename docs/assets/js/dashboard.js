// ============================================================
        // DATA LOADING
        // ============================================================
        
        let summaryData = null;
        let trajectoryData = null;
        let countryData = null;
        let cumulativeCountryData = null;
        let gdpImpactData = null;
        let modelCurves = null;
        let siteManifest = null;
        let currentSiteData = null;
        let currentSiteGeojson = null;
        let map = null;
        let isInitialMapLoad = true;
        let siteLayer = null;
        let siteGridLayer = null;
        let siteTourismLayer = null;
        let siteGridRenderer = null;
        let siteTourismRenderer = null;
        let choroplethLayer = null;
        let countryBoundaries = null;
        let renderTimeout = null;  // For debouncing
        let isRendering = false;  // Prevent concurrent renders
        let useVectorTilesForSites = false;
        const siteFileCache = new Map();
        const VALUE_TYPES = ['tourism', 'fisheries', 'coastal_protection'];
        const VALUE_TYPE_LABELS = {
            all: 'All datasets',
            tourism: 'Tourism',
            fisheries: 'Fisheries',
            coastal_protection: 'Coastal protection',
        };
        const VALUE_TYPE_DESCRIPTIONS = {
            tourism: 'Reef-associated tourism value layer.',
            fisheries: 'Reef fisheries value represented at site points.',
            coastal_protection: 'Coastal protection value represented at site points.',
        };
        
        const DATA_PATH = 'exported_data/';
        const POINT_RADIUS_CONFIG = {
            lowZoomMax: 2,  // max zoom for low radius
            midZoomMax: 7,  // max zoom for mid radius
            low: 1,  // low radius
            mid: 2,  // mid radius
            high: 4,  // high radius
        };
        // Display-only alignment tweak for point layers.
        // Set small values (e.g., +/-0.0002) only if you observe a stable offset.
        const POINT_ALIGNMENT_OFFSET = {
            lat: 0.0,
            lng: 0.0,
        };

        function getPointRadius(zoom) {
            if (zoom < POINT_RADIUS_CONFIG.lowZoomMax) return POINT_RADIUS_CONFIG.low;
            if (zoom < POINT_RADIUS_CONFIG.midZoomMax) return POINT_RADIUS_CONFIG.mid;
            return POINT_RADIUS_CONFIG.high;
        }

        function applyPointAlignmentOffset(latlng) {
            if (!latlng) return latlng;
            const dLat = Number(POINT_ALIGNMENT_OFFSET.lat || 0);
            const dLng = Number(POINT_ALIGNMENT_OFFSET.lng || 0);
            if (dLat === 0 && dLng === 0) return latlng;
            return L.latLng(latlng.lat + dLat, latlng.lng + dLng);
        }

        function lonLatToTile(lon, lat, zoom) {
            const n = Math.pow(2, zoom);
            const clampedLat = Math.max(Math.min(lat, 85.05112878), -85.05112878);
            const x = Math.max(
                0,
                Math.min(
                    n - 1,
                    Math.floor(((lon + 180) / 360) * n)
                )
            );
            const latRad = (clampedLat * Math.PI) / 180;
            const y = Math.max(
                0,
                Math.min(
                    n - 1,
                    Math.floor(
                        ((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * n
                    )
                )
            );
            return { x, y };
        }

        function getTileKeysForBounds(bounds, zoom) {
            if (!bounds) return [];
            const north = bounds.getNorth();
            const south = bounds.getSouth();
            const west = bounds.getWest();
            const east = bounds.getEast();
            const n = Math.pow(2, zoom);

            const yMin = lonLatToTile(0, north, zoom).y;
            const yMax = lonLatToTile(0, south, zoom).y;

            const lonRanges = west <= east
                ? [{ west, east }]
                : [{ west, east: 180 }, { west: -180, east }];

            const keys = new Set();
            lonRanges.forEach((range) => {
                const xMin = lonLatToTile(range.west, 0, zoom).x;
                const xMax = lonLatToTile(range.east, 0, zoom).x;
                for (let x = xMin; x <= xMax; x++) {
                    const wrappedX = ((x % n) + n) % n;
                    for (let y = yMin; y <= yMax; y++) {
                        if (y >= 0 && y < n) {
                            keys.add(`${zoom}/${wrappedX}/${y}`);
                        }
                    }
                }
            });
            return Array.from(keys);
        }

        async function fetchGeoJsonWithCache(filename) {
            if (siteFileCache.has(filename)) {
                return siteFileCache.get(filename);
            }
            const response = await fetch(DATA_PATH + filename);
            if (!response.ok) {
                return null;
            }
            const parsed = await response.json();
            siteFileCache.set(filename, parsed);
            return parsed;
        }

        /**
         * Load pre-aggregated grid-cell features for the new compact export format.
         * Geometry is stored once per value_type; metrics are columnar arrays per scenario.
         */
        async function loadGriddedSiteFeatures({
            datasetKey,
            scenarioDatasetKey,
            griddedEntry,
        }) {
            if (!griddedEntry?.grid_file || !griddedEntry?.metrics_file) {
                return [];
            }

            const [gridData, metricsData] = await Promise.all([
                fetchGeoJsonWithCache(griddedEntry.grid_file),
                fetchGeoJsonWithCache(griddedEntry.metrics_file),
            ]);

            const isPolygonGeojson =
                gridData?.type === 'FeatureCollection' && gridData?.geom_type === 'polygon';
            const hasGridCells = Array.isArray(gridData?.cells) && gridData.cells.length > 0;

            if (!isPolygonGeojson && !hasGridCells) {
                console.warn('Gridded data missing for', datasetKey, griddedEntry);
                return [];
            }
            if (!metricsData?.scenarios) {
                console.warn('Metrics data missing for', datasetKey);
                return [];
            }

            const scenarioMetrics = metricsData.scenarios[scenarioDatasetKey];
            if (!scenarioMetrics) {
                console.warn(
                    'No gridded metrics for scenario',
                    scenarioDatasetKey,
                    'available:',
                    Object.keys(metricsData.scenarios)
                );
                return [];
            }

            const resolution = Number(gridData.grid_resolution_deg) || 0.5;
            const getMetric = (name, idx) => {
                const arr = scenarioMetrics[name];
                if (!Array.isArray(arr)) return 0;
                const value = Number(arr[idx] ?? 0);
                return Number.isFinite(value) ? value : 0;
            };

            // ── Polygon path (e.g. tourism reef polygons) ──────────────────────
            if (isPolygonGeojson) {
                return gridData.features
                    .map((feature) => {
                        const ci = feature.id ?? feature.properties?.i;
                        if (ci == null) return null;
                        const valueLoss = getMetric('value_loss', ci);
                        const lossFraction = getMetric('loss_fraction', ci);
                        const originalValue = Number(feature.properties?.ov ?? 0);
                        if (originalValue <= 0 && valueLoss <= 0 && lossFraction <= 0) {
                            return null;
                        }
                        return {
                            ...feature,
                            properties: {
                                site_id: ci,
                                country: feature.properties?.co || '',
                                value_type: datasetKey,
                                original_value: originalValue,
                                n_sites: Number(feature.properties?.n ?? 1),
                                value_loss: valueLoss,
                                loss_fraction: lossFraction,
                                coral_change: getMetric('coral_change', ci),
                                annual_loss: getMetric('annual_loss', ci),
                                cumulative_loss: getMetric('cumulative_loss', ci),
                                cumulative_loss_fraction: getMetric('cumulative_loss_fraction', ci),
                            },
                        };
                    })
                    .filter(Boolean);
            }

            // ── Grid-cell path (fisheries, coastal protection) ─────────────────
            return gridData.cells
                .map((cell, idx) => {
                    const valueLoss = getMetric('value_loss', idx);
                    const lossFraction = getMetric('loss_fraction', idx);
                    const originalValue = Number(cell.ov ?? 0);
                    if (originalValue <= 0 && valueLoss <= 0 && lossFraction <= 0) {
                        return null;
                    }
                    return {
                        type: 'Feature',
                        geometry: {
                            type: 'Point',
                            coordinates: [Number(cell.lon), Number(cell.lat)],
                        },
                        properties: {
                            site_id: cell.i,
                            country: cell.co || '',
                            value_type: datasetKey,
                            original_value: originalValue,
                            n_sites: Number(cell.n ?? 1),
                            grid_resolution_deg: resolution,
                            value_loss: valueLoss,
                            loss_fraction: lossFraction,
                            coral_change: getMetric('coral_change', idx),
                            annual_loss: getMetric('annual_loss', idx),
                            cumulative_loss: getMetric('cumulative_loss', idx),
                            cumulative_loss_fraction: getMetric('cumulative_loss_fraction', idx),
                        },
                    };
                })
                .filter(Boolean);
        }

        async function loadDatasetWidePointTileFeatures({
            datasetKey,
            scenarioDatasetKey,
            tileKeys,
            datasetTileIndex,
        }) {
            const datasetEntry = datasetTileIndex?.[datasetKey];
            if (!datasetEntry) {
                return [];
            }
            const geometryFiles = datasetEntry.geometry || {};
            const attributeFiles = datasetEntry.attributes || {};
            const keysToLoad = tileKeys && tileKeys.length > 0
                ? tileKeys
                : Object.keys(geometryFiles);

            const perTileFeatures = await Promise.all(
                keysToLoad.map(async (tileKey) => {
                    const geomFile = geometryFiles[tileKey];
                    const attrFile = attributeFiles[tileKey];
                    if (!geomFile || !attrFile) {
                        return [];
                    }
                    const [geomData, attrData] = await Promise.all([
                        fetchGeoJsonWithCache(geomFile),
                        fetchGeoJsonWithCache(attrFile),
                    ]);
                    if (!geomData || !Array.isArray(geomData.features)) {
                        return [];
                    }
                    const scenarioColumns = attrData?.scenario_metrics?.[scenarioDatasetKey] || {};
                    const getMetric = (name, idx) => {
                        const arr = scenarioColumns?.[name];
                        if (!Array.isArray(arr)) return 0;
                        const value = Number(arr[idx] || 0);
                        return Number.isFinite(value) ? value : 0;
                    };
                    return geomData.features.map((feature, idx) => ({
                        type: 'Feature',
                        geometry: feature.geometry,
                        properties: {
                            ...(feature.properties || {}),
                            value_loss: getMetric('value_loss', idx),
                            loss_fraction: getMetric('loss_fraction', idx),
                            coral_change: getMetric('coral_change', idx),
                            annual_loss: getMetric('annual_loss', idx),
                            cumulative_loss: getMetric('cumulative_loss', idx),
                            cumulative_loss_fraction: getMetric('cumulative_loss_fraction', idx),
                        },
                    }));
                })
            );

            return perTileFeatures.flat();
        }

        function getSelectedValueType(controlId, fallback = 'all') {
            const control = document.getElementById(controlId);
            return control ? control.value : fallback;
        }

        function formatValueType(valueType) {
            return VALUE_TYPE_LABELS[valueType] || valueType;
        }

        function describeValueType(valueType) {
            return VALUE_TYPE_DESCRIPTIONS[valueType] || 'Reef-associated economic value layer.';
        }

        function getScenarioComparisonValueTypes() {
            const checkboxes = document.querySelectorAll('.scenario-value-type-checkbox');
            const selected = Array.from(checkboxes)
                .filter((cb) => cb.checked)
                .map((cb) => cb.value);
            return selected;
        }

        function getMapSelectedValueTypes() {
            const checkboxes = document.querySelectorAll('.map-value-type-checkbox');
            return Array.from(checkboxes)
                .filter((cb) => cb.checked)
                .map((cb) => cb.value);
        }

        function getTrajectorySelectedValueTypes() {
            const checkboxes = document.querySelectorAll('.traj-value-type-checkbox');
            return Array.from(checkboxes)
                .filter((cb) => cb.checked)
                .map((cb) => cb.value);
        }

        function formatBillions(valueInBillions) {
            const abs = Math.abs(Number(valueInBillions || 0));
            if (abs >= 1) return `$${valueInBillions.toFixed(2)}B`;
            return `$${(valueInBillions * 1000).toFixed(1)}M`;
        }

        function aggregateRowsByCountry(rows, isCumulative = false, includeGdp = false) {
            const grouped = new Map();
            rows.forEach((row) => {
                const country = row.country || '';
                const key = `${country}||${row.iso_a3 || ''}`;
                if (!grouped.has(key)) {
                    grouped.set(key, {
                        country,
                        iso_a3: row.iso_a3 || '',
                        original_value: 0,
                        value_loss: 0,
                        cumulative_loss: 0,
                        annual_loss: 0,
                        loss_fraction: 0,
                        cumulative_loss_fraction: 0,
                        _annual_fraction_weighted_loss: 0,
                        _cumulative_fraction_weighted_loss: 0,
                        _gdp_loss: 0,
                        _gdp_base: null,
                    });
                }
                const acc = grouped.get(key);
                acc.original_value += Number(row.original_value || 0);
                acc.value_loss += Number(row.value_loss || 0);
                acc.cumulative_loss += Number(row.cumulative_loss || 0);
                acc.annual_loss += Number(row.annual_loss || row.value_loss || 0);
                acc._annual_fraction_weighted_loss += Number(row.original_value || 0) * Number(row.loss_fraction || 0);
                acc._cumulative_fraction_weighted_loss += Number(row.original_value || 0) * Number(row.cumulative_loss_fraction || 0);

                if (includeGdp) {
                    const gdpKey = `${row.scenario}||${row.model}||${row.value_type || ''}||${country}`;
                    const gdpRec = window.gdpImpactLookup ? window.gdpImpactLookup[gdpKey] : null;
                    if (gdpRec) {
                        acc._gdp_loss += Number(gdpRec.value_loss || 0);
                        const nationalGdp = Number(gdpRec.national_gdp || 0);
                        if (nationalGdp > 0) {
                            acc._gdp_base = nationalGdp;
                        }
                    }
                }
            });

            return Array.from(grouped.values()).map((row) => {
                const original = row.original_value;
                row.loss_fraction = original > 0
                    ? (row._annual_fraction_weighted_loss > 0
                        ? row._annual_fraction_weighted_loss / original
                        : row.value_loss / original)
                    : 0;
                row.cumulative_loss_fraction = original > 0
                    ? (row._cumulative_fraction_weighted_loss > 0
                        ? row._cumulative_fraction_weighted_loss / original
                        : row.cumulative_loss / original)
                    : 0;
                if (includeGdp) {
                    row.loss_as_gdp_pct = row._gdp_base && row._gdp_base > 0
                        ? (row._gdp_loss / row._gdp_base) * 100
                        : 0;
                }
                delete row._annual_fraction_weighted_loss;
                delete row._cumulative_fraction_weighted_loss;
                delete row._gdp_loss;
                delete row._gdp_base;
                return row;
            });
        }
        
        async function loadData() {
            try {
                const [summary, trajectories, countries, cumulativeCountries, curves, manifest, gdpImpacts] = await Promise.all([
                    fetch(DATA_PATH + 'summary.json').then(r => r.json()),
                    fetch(DATA_PATH + 'trajectories.json').then(r => r.json()),
                    fetch(DATA_PATH + 'country_results.json').then(r => r.json()),
                    fetch(DATA_PATH + 'cumulative_country_results.json').then(r => r.json()).catch(() => []),
                    fetch(DATA_PATH + 'model_curves.json').then(r => r.json()),
                    fetch(DATA_PATH + 'manifest.json').then(r => r.json()),
                    fetch(DATA_PATH + 'gdp_impacts.json').then(r => r.json()).catch(() => null),
                ]);
                
                summaryData = summary;
                trajectoryData = trajectories;
                countryData = countries;
                cumulativeCountryData = cumulativeCountries;
                gdpImpactData = gdpImpacts;
                // Build lookup for GDP impact by (scenario, model, country)
                window.gdpImpactLookup = {};
                if (Array.isArray(gdpImpactData)) {
                    gdpImpactData.forEach(d => {
                        const key = `${d.scenario}||${d.model}||${d.value_type || ''}||${d.country}`;
                        window.gdpImpactLookup[key] = d;
                    });
                }
                modelCurves = curves;
                siteManifest = manifest;
                
                console.log('Data loaded successfully:', {
                    summary: !!summary,
                    trajectories: trajectories?.length || 0,
                    countries: countries?.length || 0,
                    cumulativeCountries: cumulativeCountries?.length || 0,
                    gdpImpacts: !!gdpImpacts,
                    curves: !!curves,
                    manifest: !!manifest
                });
                
                if (cumulativeCountries && cumulativeCountries.length > 0) {
                    const uniqueScenarios = [...new Set(cumulativeCountries.map(c => c.scenario))];
                    const uniqueModels = [...new Set(cumulativeCountries.map(c => c.model))];
                    console.log('Cumulative data loaded:', {
                        totalRecords: cumulativeCountries.length,
                        uniqueScenarios,
                        uniqueModels
                    });
                } else {
                    console.warn('No cumulative country data loaded');
                }
                
                initializeDashboard();
            } catch (error) {
                console.error('Error loading data:', error);
                document.getElementById('summary-stats').innerHTML = 
                    '<p style="color: var(--accent-red);">Error loading data. Please run the export script first.</p>';
            }
        }
        
        // ============================================================
        // INITIALIZATION
        // ============================================================
        
        function initializeDashboard() {
            renderSummaryStats();
            renderScenarioComparison();
            renderOverviewTrajectory('cumulative_loss');
            renderModelComparison();
            initializeMap();
            
            // Set up event listeners
            setupNavigation();
            setupControls();
            
            // Load site data if map page is active
            setTimeout(() => {
                const mapPage = document.getElementById('page-map');
                if (mapPage && mapPage.classList.contains('active')) {
                    loadSiteData();
                }
            }, 500);
        }
        
        function setupNavigation() {
            const menuToggle = document.getElementById('nav-menu-toggle');
            const mobileMenu = document.getElementById('nav-mobile-menu');
            
            // Toggle mobile menu
            if (menuToggle && mobileMenu) {
                menuToggle.addEventListener('click', () => {
                    menuToggle.classList.toggle('active');
                    mobileMenu.classList.toggle('active');
                });
                
                // Close mobile menu when clicking outside
                document.addEventListener('click', (e) => {
                    if (!menuToggle.contains(e.target) && !mobileMenu.contains(e.target)) {
                        menuToggle.classList.remove('active');
                        mobileMenu.classList.remove('active');
                    }
                });
            }
            
            // Handle navigation clicks
            document.querySelectorAll('.nav-link').forEach(link => {
                link.addEventListener('click', (e) => {
                    const page = e.target.dataset.page;
                    
                    // Close mobile menu
                    if (menuToggle && mobileMenu) {
                        menuToggle.classList.remove('active');
                        mobileMenu.classList.remove('active');
                    }
                    
                    // Update nav
                    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                    e.target.classList.add('active');
                    
                    // Show page
                    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
                    document.getElementById('page-' + page).classList.add('active');
                    
                    // Initialize page-specific content
                    if (page === 'map') {
                        setTimeout(() => {
                            if (map) map.invalidateSize();
                            loadSiteData();
                        }, 100);
                    } else if (page === 'trajectories') {
                        renderTrajectoryPage();
                    } else if (page === 'gdp') {
                        renderCountryChart();
                        renderGdpComparison();
                    }
                });
            });
        }
        
        function setupControls() {
            // Overview trajectory toggle
            document.querySelectorAll('.toggle-btn[data-metric]').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    document.querySelectorAll('.toggle-btn[data-metric]').forEach(b => b.classList.remove('active'));
                    e.target.classList.add('active');
                    const currentMetric = e.target.dataset.metric;
                    renderOverviewTrajectory(currentMetric);
                });
            });
            
            // Overview model selector
            document.getElementById('overview-model').addEventListener('change', () => {
                const activeBtn = document.querySelector('.toggle-btn[data-metric].active');
                const metric = activeBtn ? activeBtn.dataset.metric : 'cumulative_loss';
                renderOverviewTrajectory(metric);
            });
            document.getElementById('overview-value-type').addEventListener('change', () => {
                renderSummaryStats();
                renderScenarioComparison();
                const activeBtn = document.querySelector('.toggle-btn[data-metric].active');
                const metric = activeBtn ? activeBtn.dataset.metric : 'cumulative_loss';
                renderOverviewTrajectory(metric);
            });
            document.querySelectorAll('.scenario-value-type-checkbox').forEach((checkbox) => {
                checkbox.addEventListener('change', () => {
                    renderScenarioComparison();
                });
            });
            
            // Map controls
            // Debounce filter changes to avoid excessive re-renders
            const debouncedLoadSiteData = () => {
                clearTimeout(renderTimeout);
                renderTimeout = setTimeout(() => {
                    loadSiteData();
                }, 200);  // 200ms debounce
            };
            
            document.getElementById('map-scenario').addEventListener('change', debouncedLoadSiteData);
            document.getElementById('map-model').addEventListener('change', debouncedLoadSiteData);
            document.querySelectorAll('.map-value-type-checkbox').forEach((checkbox) => {
                checkbox.addEventListener('change', debouncedLoadSiteData);
            });
            document.getElementById('map-metric').addEventListener('change', () => {
                // Debounce metric changes
                clearTimeout(renderTimeout);
                renderTimeout = setTimeout(() => {
                    // Re-render sites and choropleth with new metric
                    if (currentSiteGeojson) {
                        // Preserve current map view when changing metric
                        let currentView = null;
                        if (map) {
                            currentView = {
                                center: map.getCenter(),
                                zoom: map.getZoom()
                            };
                        }
                        const scenario = document.getElementById('map-scenario').value;
                        const isCumulative = scenario.startsWith('cumulative_');
                        renderSites(currentSiteGeojson, isCumulative, currentView);
                    }
                    if (document.getElementById('map-choropleth-toggle').checked) {
                        renderChoropleth();
                    }
                }, 150);
            });
            document.getElementById('map-choropleth-toggle').addEventListener('change', toggleChoropleth);
            
            // Trajectory controls
            document.getElementById('traj-interpolation').addEventListener('change', renderTrajectoryPage);
            document.getElementById('traj-model').addEventListener('change', renderTrajectoryPage);
            document.querySelectorAll('.traj-value-type-checkbox').forEach((checkbox) => {
                checkbox.addEventListener('change', renderTrajectoryPage);
            });
            
            // Country controls
            document.getElementById('country-scenario').addEventListener('change', renderCountryChart);
            document.getElementById('country-model').addEventListener('change', renderCountryChart);
            document.getElementById('country-value-type').addEventListener('change', renderCountryChart);
            document.getElementById('country-limit').addEventListener('change', renderCountryChart);
            document.getElementById('country-metric').addEventListener('change', renderCountryChart);
            document.getElementById('country-color-mode').addEventListener('change', renderCountryChart);
            
            // GDP comparison controls
            document.getElementById('gdp-comparison-model').addEventListener('change', renderGdpComparison);
            document.getElementById('gdp-value-type').addEventListener('change', renderGdpComparison);
            document.getElementById('gdp-comparison-metric').addEventListener('change', renderGdpComparison);
            document.getElementById('gdp-comparison-limit').addEventListener('change', renderGdpComparison);
        }
        
        // ============================================================
        // RENDERING FUNCTIONS
        // ============================================================
        
        function renderSummaryStats() {
            if (!summaryData) return;
            const snapshotRows = summaryData.snapshot_results || [];
            const cumulativeRows = summaryData.cumulative_results || [];
            if (!snapshotRows.length || !cumulativeRows.length) return;

            const getBaseline = (valueType) => {
                const rows = snapshotRows.filter((r) => r.value_type === valueType);
                if (!rows.length) return 0;
                return Math.max(...rows.map((r) => Number(r.original_value_billions || 0)));
            };

            const baselineTourism = getBaseline('tourism');
            const baselineFisheries = getBaseline('fisheries');
            const baselineCoastal = getBaseline('coastal_protection');

            const annualAggregate = new Map();
            snapshotRows.forEach((r) => {
                const key = `${r.scenario}||${r.model}`;
                annualAggregate.set(
                    key,
                    (annualAggregate.get(key) || 0) + Number(r.total_loss_billions || 0)
                );
            });
            let worstAnnualLoss = 0;
            annualAggregate.forEach((value) => {
                if (value > worstAnnualLoss) worstAnnualLoss = value;
            });

            const cumulativeAggregate = new Map();
            cumulativeRows.forEach((r) => {
                const key = `${r.scenario}||${r.interpolation}||${r.model}`;
                cumulativeAggregate.set(
                    key,
                    (cumulativeAggregate.get(key) || 0) + Number(r.total_cumulative_loss_trillions || 0)
                );
            });
            let worstCumulativeLoss = 0;
            cumulativeAggregate.forEach((value) => {
                if (value > worstCumulativeLoss) worstCumulativeLoss = value;
            });

            const coralRow =
                cumulativeRows.find((r) => {
                    const scenario = (r.scenario || '').toLowerCase();
                    return scenario.includes('rcp85') && (r.period || '').includes('2100') && r.interpolation === 'linear';
                }) ||
                cumulativeRows.find((r) => (r.scenario || '').toLowerCase().includes('rcp85')) ||
                cumulativeRows[0];
            if (!coralRow) return;

            const html = `
                <div class="summary-row summary-row-top" style="text-align: center;">
                    <div class="stat-card">
                        <div class="stat-label">Coral Cover Change</div>
                        <div class="stat-value neutral">${Number(coralRow.cover_change_pp || 0).toFixed(1)}pp</div>
                        <div class="stat-detail">${Number(coralRow.baseline_cover_pct || 0).toFixed(1)}% → ${Number(coralRow.final_cover_pct || 0).toFixed(1)}%</div>
                    </div>
                </div>
                <div class="summary-row summary-row-middle">
                    <div class="stat-card">
                        <div class="stat-label">Baseline Tourism Value</div>
                        <div class="stat-value value">${formatBillions(baselineTourism)}</div>
                        <div class="stat-detail">Annual reef-associated tourism</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Baseline Coastal Protection Value</div>
                        <div class="stat-value value">${formatBillions(baselineCoastal)}</div>
                        <div class="stat-detail">Annual reef-derived flood protection</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Baseline Fisheries Value</div>
                        <div class="stat-value value">${formatBillions(baselineFisheries)}</div>
                        <div class="stat-detail">Annual reef fisheries value</div>
                    </div>
                </div>
                <div class="summary-row summary-row-bottom">
                    <div class="stat-card">
                        <div class="stat-label">Worst-Case Aggregate Annual Loss</div>
                        <div class="stat-value loss">-${formatBillions(worstAnnualLoss)}</div>
                        <div class="stat-detail">Summed across all three datasets</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Worst-Case Aggregate Cumulative Loss</div>
                        <div class="stat-value loss">-$${Number(worstCumulativeLoss || 0).toFixed(2)}T</div>
                        <div class="stat-detail">2013 onward, summed across datasets</div>
                    </div>
                </div>
            `;
            
            document.getElementById('summary-stats').innerHTML = html;
        }
        
        function renderScenarioComparison() {
            if (!summaryData) return;
            const selectedValueTypes = getScenarioComparisonValueTypes();
            if (selectedValueTypes.length === 0) {
                Plotly.newPlot(
                    'scenario-comparison-chart',
                    [],
                    {
                        paper_bgcolor: 'transparent',
                        plot_bgcolor: 'transparent',
                        font: { color: '#94a3b8', family: 'Instrument Sans' },
                        xaxis: { gridcolor: '#334155', title: 'Scenario' },
                        yaxis: { gridcolor: '#334155', title: 'Annual Loss ($ Billion)' },
                        annotations: [
                            {
                                x: 0.5,
                                y: 0.5,
                                xref: 'paper',
                                yref: 'paper',
                                text: 'Select a dataset to see its values',
                                showarrow: false,
                                font: { size: 16, color: '#94a3b8' },
                            },
                        ],
                        margin: { t: 20, r: 20, b: 100, l: 60 },
                    },
                    { responsive: true }
                );
                return;
            }
            
            let results = summaryData.snapshot_results || [];
            results = results.filter((r) => selectedValueTypes.includes(r.value_type));
            const grouped = new Map();
            results.forEach((r) => {
                const key = `${r.scenario}||${r.model}`;
                if (!grouped.has(key)) {
                    grouped.set(key, {
                        ...r,
                        value_type: 'aggregate_selected',
                        original_value_billions: 0,
                        remaining_value_billions: 0,
                        total_loss_billions: 0,
                        loss_fraction_pct: 0,
                    });
                }
                const g = grouped.get(key);
                g.original_value_billions += Number(r.original_value_billions || 0);
                g.remaining_value_billions += Number(r.remaining_value_billions || 0);
                g.total_loss_billions += Number(r.total_loss_billions || 0);
                g.loss_fraction_pct = g.original_value_billions > 0
                    ? (g.total_loss_billions / g.original_value_billions) * 100
                    : 0;
            });
            results = Array.from(grouped.values());
            if (!results.length) return;
            
            const models = [...new Set(results.map(r => r.model))];
            
            // Map ugly scenario names to nice labels
            const formatScenario = (s) => {
                const match = s.match(/rcp(\d+)_(\d+)/i);
                if (match) {
                    return `RCP ${match[1].charAt(0)}.${match[1].charAt(1)} - ${match[2]}`;
                }
                return s.replace('y_future_', '').replace(/_/g, ' ').toUpperCase();
            };
            
            // Distinct colors for each model
            const modelColors = {
                'Tipping Point (threshold=10%)': '#F11B00',
                'Compound (3.81%/pp)': '#3A9AB2',
                'Linear (3.81%/pp)': '#E3B710',
            };
            
            const traces = models.map(model => ({
                x: results.filter(r => r.model === model).map(r => formatScenario(r.scenario)),
                y: results.filter(r => r.model === model).map(r => r.total_loss_billions),
                name: model.includes('Linear') ? 'Linear' : 
                      model.includes('Compound') ? 'Compound' : 'Tipping Point',
                type: 'bar',
                hovertemplate:
                    'Scenario: %{x}<br>' +
                    'Model: %{fullData.name}<br>' +
                    'Annual Loss: $%{y:.1f}B<extra></extra>',
                marker: {
                    color: modelColors[model] || '#94a3b8'
                }
            }));
            
            const layout = {
                barmode: 'group',
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                xaxis: { 
                    gridcolor: '#334155',
                    title: 'Scenario',
                    tickangle: 0
                },
                yaxis: { 
                    gridcolor: '#334155',
                    title: 'Annual Loss ($ Billion)'
                },
                legend: { 
                    orientation: 'h', 
                    y: -0.2,
                    font: { size: 12 }
                },
                margin: { t: 20, r: 20, b: 100, l: 60 }
            };
            
            Plotly.newPlot('scenario-comparison-chart', traces, layout, {responsive: true});
        }
        
        function renderOverviewTrajectory(metric) {
            if (!trajectoryData) return;
            
            const modelFilter = document.getElementById('overview-model').value;
            const selectedValueType = getSelectedValueType('overview-value-type', 'all');
            
            // Color and linestyle config - matching scenario comparison chart
            // Color = RCP scenario (blue for RCP45, red for RCP85)
            const rcpColors = {
                'rcp45': '#3498db',  // Blue (matching scenario comparison)
                'rcp85': '#e74c3c',  // Red (matching scenario comparison)
            };
            // Linestyle = dataset type
            const datasetLineStyles = {
                tourism: 'solid',
                fisheries: 'dash',
                coastal_protection: 'dot',
            };

            // Map trace types for Plotly
            const metricMap = {
                'cumulative_loss': { title: 'Cumulative Loss ($ Trillion)' },
                'annual_loss': { title: 'Annual Loss ($ Billion/year)' }
            };
            const config = metricMap[metric];

            // Filter data: use linear interpolation only, apply model filter
            let filtered = trajectoryData.filter(t => t.interpolation === 'linear');
            if (selectedValueType !== 'all') {
                filtered = filtered.filter(t => t.value_type === selectedValueType);
            }
            if (modelFilter !== 'all') {
                filtered = filtered.filter(t => {
                    const modelName = t.model.toLowerCase();
                    if (modelFilter === 'Linear') {
                        return modelName.includes('linear') && !modelName.includes('compound') && !modelName.includes('tipping');
                    } else if (modelFilter === 'Compound') {
                        return modelName.includes('compound') && !modelName.includes('tipping');
                    } else if (modelFilter === 'Tipping') {
                        return modelName.includes('tipping');
                    }
                    return t.model.includes(modelFilter);
                });
            }

            // Create traces
            const traces = filtered.map(t => {
                const scenario = t.scenario.toLowerCase();
                const color = rcpColors[scenario] || '#94a3b8';
                const linestyle = datasetLineStyles[t.value_type] || 'solid';
                
                let yData;
                if (metric === 'cumulative_loss') {
                    // Cumulative loss: use cumulative_losses directly (already in trillions from export)
                    // This is the cumulative sum of opportunity cost
                    yData = t.cumulative_loss || [];
                } else if (metric === 'annual_loss') {
                    // Annual loss: year-on-year value lost (value_lost_this_year)
                    // This is the year-over-year decline in value (already in billions from export)
                    yData = t.annual_value_lost || t.annual_loss || [];
                }
                
                return {
                    x: t.years,
                    y: yData,
                    name: `${t.scenario.toUpperCase()} - ${formatValueType(t.value_type)}`,
                    showlegend: false,
                    mode: 'lines',
                    hovertemplate:
                        `${t.scenario.toUpperCase()} | ${formatValueType(t.value_type)}<br>` +
                        'Year: %{x}<br>' +
                        `${metric === 'cumulative_loss' ? 'Cumulative Loss: $%{y:.3f}T' : 'Annual Loss: $%{y:.2f}B/yr'}<extra></extra>`,
                    line: {
                        color: color,
                        dash: linestyle,
                        width: 2.5
                    }
                };
            });

            // Legend guide: colors indicate scenario, dash indicates dataset
            const legendGuideTraces = [
                {
                    x: [null], y: [null], mode: 'lines', name: 'RCP 4.5',
                    line: { color: rcpColors.rcp45, dash: 'solid', width: 3 },
                    hoverinfo: 'skip'
                },
                {
                    x: [null], y: [null], mode: 'lines', name: 'RCP 8.5',
                    line: { color: rcpColors.rcp85, dash: 'solid', width: 3 },
                    hoverinfo: 'skip'
                },
                {
                    x: [null], y: [null], mode: 'lines', name: 'Tourism',
                    line: { color: '#cbd5e1', dash: datasetLineStyles.tourism, width: 3 },
                    hoverinfo: 'skip'
                },
                {
                    x: [null], y: [null], mode: 'lines', name: 'Fisheries',
                    line: { color: '#cbd5e1', dash: datasetLineStyles.fisheries, width: 3 },
                    hoverinfo: 'skip'
                },
                {
                    x: [null], y: [null], mode: 'lines', name: 'Coastal protection',
                    line: { color: '#cbd5e1', dash: datasetLineStyles.coastal_protection, width: 3 },
                    hoverinfo: 'skip'
                },
            ];
            
            const layout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                xaxis: { 
                    gridcolor: '#334155',
                    title: 'Year'
                },
                yaxis: { 
                    gridcolor: '#334155',
                    title: config.title,
                    // Adjust scale based on metric
                    type: metric === 'cumulative_loss' ? 'linear' : 'linear'
                },
                legend: { 
                    orientation: 'h', 
                    y: -0.25, // Move legend further down below plot, increasing space from x axis label
                    font: { size: 11 },
                },
                margin: { t: 20, r: 20, b: 80, l: 60 }
            };
            
            Plotly.newPlot('trajectory-chart', [...traces, ...legendGuideTraces], layout, {responsive: true});
        }
        
        function renderTrajectoryPage() {
            if (!trajectoryData) return;
            
            const interpolation = document.getElementById('traj-interpolation').value;
            const modelFilter = document.getElementById('traj-model').value;
            const selectedValueTypes = getTrajectorySelectedValueTypes();
            
            // Color = RCP scenario (blue for RCP45, red for RCP85) - matching scenario comparison chart
            const rcpColors = {
                'rcp45': '#3498db',  // Blue
                'rcp85': '#e74c3c',  // Red
            };
            // Linestyle = dataset
            const datasetLineStyles = {
                tourism: 'solid',
                fisheries: 'dash',
                coastal_protection: 'dot',
            };
            
            // For coral cover, we only need unique scenario+interpolation combos (not per model)
            // since coral cover is the same for all economic models
            // Use a Set to get unique scenario+interpolation combos
            const seenCoralKeys = new Set();
            const coralFiltered = trajectoryData.filter(t => {
                const interpMatch = t.interpolation === interpolation;
                if (!interpMatch) return false;
                
                const key = `${t.scenario}_${t.interpolation}`;
                if (seenCoralKeys.has(key)) return false;
                seenCoralKeys.add(key);
                return true;
            });
            
            const coralTraces = coralFiltered.map(t => ({
                x: t.years,
                y: t.coral_cover,
                name: `${t.scenario.toUpperCase()}`,
                mode: 'lines',
                hovertemplate:
                    `${t.scenario.toUpperCase()}<br>` +
                    'Year: %{x}<br>' +
                    'Coral Cover: %{y:.1f}%<extra></extra>',
                line: { 
                    color: rcpColors[t.scenario.toLowerCase()] || '#94a3b8', 
                    width: 3 
                }
            }));
            
            Plotly.newPlot('coral-cover-chart', coralTraces, {
                paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                xaxis: { gridcolor: '#334155', title: 'Year' },
                yaxis: { gridcolor: '#334155', title: 'Coral Cover (%)' },
                legend: { orientation: 'h', y: -0.22, font: { size: 10 } },
                margin: { t: 20, r: 20, b: 90, l: 60 }
            }, {responsive: true});
            
            // For economic charts, filter by both interpolation and model
            let economicFiltered = trajectoryData.filter(t => t.interpolation === interpolation);
            if (modelFilter !== 'all') {
                economicFiltered = economicFiltered.filter(t => t.model.includes(modelFilter));
            }
            economicFiltered = economicFiltered.filter(t => selectedValueTypes.includes(t.value_type));

            if (selectedValueTypes.length === 0) {
                const emptyLayout = (title) => ({
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    font: { color: '#94a3b8', family: 'Instrument Sans' },
                    xaxis: { gridcolor: '#334155', title: 'Year' },
                    yaxis: { gridcolor: '#334155', title },
                    annotations: [
                        {
                            x: 0.5,
                            y: 0.5,
                            xref: 'paper',
                            yref: 'paper',
                            text: 'Select at least one dataset',
                            showarrow: false,
                            font: { size: 15, color: '#94a3b8' },
                        },
                    ],
                    margin: { t: 20, r: 20, b: 60, l: 60 },
                });
                Plotly.newPlot('annual-value-chart', [], emptyLayout('Annual Value ($ Billion)'), {responsive: true});
                Plotly.newPlot('cumulative-chart', [], emptyLayout('Cumulative Loss ($ Trillion)'), {responsive: true});
                return;
            }
            
            // Annual value chart - distinguish by model and scenario
            // Color = RCP scenario, Linestyle = Model
            const valueTraces = economicFiltered.map(t => {
                const scenario = t.scenario.toLowerCase();
                const color = rcpColors[scenario] || '#94a3b8';
                const dash = datasetLineStyles[t.value_type] || 'solid';
                
                return {
                    x: t.years,
                    y: t.annual_value,
                    name: `${t.scenario.toUpperCase()} - ${formatValueType(t.value_type)}`,
                    showlegend: false,
                    mode: 'lines',
                    hovertemplate:
                        `${t.scenario.toUpperCase()} | ${formatValueType(t.value_type)}<br>` +
                        'Year: %{x}<br>' +
                        'Annual Value: $%{y:.2f}B<extra></extra>',
                    line: { 
                        color: color,
                        dash: dash,
                        width: 2.5 
                    }
                };
            });

            const trajectoryLegendGuideTraces = [
                {
                    x: [null], y: [null], mode: 'lines', name: 'RCP 4.5',
                    line: { color: rcpColors.rcp45, dash: 'solid', width: 3 },
                    hoverinfo: 'skip'
                },
                {
                    x: [null], y: [null], mode: 'lines', name: 'RCP 8.5',
                    line: { color: rcpColors.rcp85, dash: 'solid', width: 3 },
                    hoverinfo: 'skip'
                },
                {
                    x: [null], y: [null], mode: 'lines', name: 'Tourism',
                    line: { color: '#cbd5e1', dash: datasetLineStyles.tourism, width: 3 },
                    hoverinfo: 'skip'
                },
                {
                    x: [null], y: [null], mode: 'lines', name: 'Fisheries',
                    line: { color: '#cbd5e1', dash: datasetLineStyles.fisheries, width: 3 },
                    hoverinfo: 'skip'
                },
                {
                    x: [null], y: [null], mode: 'lines', name: 'Coastal protection',
                    line: { color: '#cbd5e1', dash: datasetLineStyles.coastal_protection, width: 3 },
                    hoverinfo: 'skip'
                },
            ];
            
            Plotly.newPlot('annual-value-chart', [...valueTraces, ...trajectoryLegendGuideTraces], {
                paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                xaxis: { gridcolor: '#334155', title: 'Year' },
                yaxis: { gridcolor: '#334155', title: 'Annual Value ($ Billion)' },
                legend: { orientation: 'h', y: -0.28, font: { size: 10 } },
                margin: { t: 20, r: 20, b: 100, l: 60 }
            }, {responsive: true            });
            
            // Cumulative loss chart - cumulative sum of opportunity cost
            // Color = RCP scenario, Linestyle = Model
            const cumulativeTraces = economicFiltered.map(t => {
                // Calculate cumulative sum of opportunity cost
                const oppCost = t.annual_opportunity_cost || [];
                let cumulative = 0;
                const cumulativeOppCost = oppCost.length > 0 
                    ? oppCost.map(val => {
                        cumulative += val;
                        return cumulative / 1e3; // Convert to trillions
                    })
                    : (t.cumulative_loss || []); // Fallback for old data
                
                const scenario = t.scenario.toLowerCase();
                const color = rcpColors[scenario] || '#94a3b8';
                const dash = datasetLineStyles[t.value_type] || 'solid';
                
                return {
                    x: t.years,
                    y: cumulativeOppCost,
                    name: `${t.scenario.toUpperCase()} - ${formatValueType(t.value_type)}`,
                    showlegend: false,
                    mode: 'lines',
                    hovertemplate:
                        `${t.scenario.toUpperCase()} | ${formatValueType(t.value_type)}<br>` +
                        'Year: %{x}<br>' +
                        'Cumulative Loss: $%{y:.3f}T<extra></extra>',
                    line: { 
                        color: color,
                        dash: dash,
                        width: 2.5 
                    }
                };
            });
            
            Plotly.newPlot('cumulative-chart', [...cumulativeTraces, ...trajectoryLegendGuideTraces], {
                paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                xaxis: { gridcolor: '#334155', title: 'Year' },
                yaxis: { gridcolor: '#334155', title: 'Cumulative Loss ($ Trillion)' },
                legend: { orientation: 'h', y: -0.28, font: { size: 10 } },
                margin: { t: 20, r: 20, b: 100, l: 60 }
            }, {responsive: true});
        }
        
        function renderCountryChart() {
            const scenario = document.getElementById('country-scenario').value;
            const isCumulative = scenario.startsWith('cumulative_');
            const dataSource = isCumulative ? cumulativeCountryData : countryData;
            const valueType = getSelectedValueType('country-value-type', 'all');
            
            if (!dataSource) {
                console.warn('Country data not loaded. isCumulative:', isCumulative, 'dataSource:', !!dataSource);
                return;
            }
            
            const model = document.getElementById('country-model').value;
            const limitValue = document.getElementById('country-limit').value;
            const metric = document.getElementById('country-metric').value;
            const colorMode = document.getElementById('country-color-mode').value;
            
            console.log('Rendering country chart:', { scenario, model, isCumulative, dataSourceLength: dataSource.length });
            
            // Match scenario exactly
            let filtered = dataSource.filter(c => 
                c.scenario === scenario && c.model === model
            );
            if (valueType !== 'all') {
                filtered = filtered.filter(c => c.value_type === valueType);
            } else {
                filtered = aggregateRowsByCountry(
                    filtered.map(row => ({ ...row, scenario, model })),
                    isCumulative,
                    true
                );
            }
            
            console.log('Filtered countries:', filtered.length, 'for scenario:', scenario, 'model:', model);
            
            // Sort by selected metric
            if (metric === 'value_loss') {
                const lossKey = isCumulative ? 'cumulative_loss' : 'value_loss';
                filtered = filtered.sort((a, b) => (b[lossKey] || 0) - (a[lossKey] || 0));
            } else if (metric === 'reef_loss_pct') {
                const fractionKey = isCumulative ? 'cumulative_loss_fraction' : 'loss_fraction';
                filtered = filtered.sort((a, b) => (b[fractionKey] || 0) - (a[fractionKey] || 0));
            } else if (metric === 'gdp_pct' && !isCumulative && window.gdpImpactLookup) {
                // Sort by GDP % if available
                filtered = filtered.sort((a, b) => {
                    const valA = valueType === 'all'
                        ? (a.loss_as_gdp_pct || 0)
                        : ((window.gdpImpactLookup[`${scenario}||${model}||${valueType}||${a.country}`] || {}).loss_as_gdp_pct || 0);
                    const valB = valueType === 'all'
                        ? (b.loss_as_gdp_pct || 0)
                        : ((window.gdpImpactLookup[`${scenario}||${model}||${valueType}||${b.country}`] || {}).loss_as_gdp_pct || 0);
                    return valB - valA;
                });
            } else {
                // Fallback: sort by reef loss fraction
                const fractionKey = isCumulative ? 'cumulative_loss_fraction' : 'loss_fraction';
                filtered = filtered.sort((a, b) => (b[fractionKey] || 0) - (a[fractionKey] || 0));
            }
            
            // Apply limit
            if (limitValue !== 'all') {
                filtered = filtered.slice(0, parseInt(limitValue));
            }
            
            // Dynamic height based on number of countries
            const chartHeight = Math.max(400, filtered.length * 25);
            document.getElementById('country-chart').style.height = chartHeight + 'px';
            
            // Configure based on metric (x-axis)
            let xData, xTitle, textFn;
            if (metric === 'value_loss') {
                const lossKey = isCumulative ? 'cumulative_loss' : 'value_loss';
                xData = filtered.map(c => (c[lossKey] || 0) / 1e6);
                xTitle = isCumulative ? 'Cumulative Loss ($ Million)' : 'Value Loss ($ Million)';
                textFn = c => `$${((c[lossKey] || 0) / 1e6).toFixed(1)}M`;
            } else if (metric === 'reef_loss_pct') {
                const fractionKey = isCumulative ? 'cumulative_loss_fraction' : 'loss_fraction';
                xData = filtered.map(c => (c[fractionKey] || 0) * 100);
                xTitle = isCumulative ? 'Cumulative Loss Fraction (%)' : 'Loss Fraction (%)';
                textFn = c => `${((c[fractionKey] || 0) * 100).toFixed(1)}%`;
            } else if (metric === 'gdp_pct') {
                // For GDP %, need to look up from gdpImpactData
                if (!isCumulative && window.gdpImpactLookup) {
                    xData = filtered.map(c => {
                        if (valueType === 'all') return c.loss_as_gdp_pct || 0;
                        const rec = window.gdpImpactLookup[`${scenario}||${model}||${valueType}||${c.country}`];
                        return rec ? rec.loss_as_gdp_pct : 0;
                    });
                    xTitle = 'Loss as % of National GDP';
                    textFn = c => {
                        if (valueType === 'all') return `${(c.loss_as_gdp_pct || 0).toFixed(2)}%`;
                        const rec = window.gdpImpactLookup[`${scenario}||${model}||${valueType}||${c.country}`];
                        return rec ? `${rec.loss_as_gdp_pct.toFixed(2)}%` : '0%';
                    };
                } else {
                    // Fallback for cumulative or if GDP data not available
                    const fractionKey = isCumulative ? 'cumulative_loss_fraction' : 'loss_fraction';
                    xData = filtered.map(c => (c[fractionKey] || 0) * 100);
                    xTitle = isCumulative ? 'Cumulative Loss Fraction (%)' : 'Loss Fraction (%)';
                    textFn = c => `${((c[fractionKey] || 0) * 100).toFixed(1)}%`;
                }
            }

            // Configure colouring (loss % of reef tourism, % of national GDP, or absolute value loss)
            let colorValues;
            let colorbarTitle;
            let colorbarFormat;

            const reefFractionKey = isCumulative ? 'cumulative_loss_fraction' : 'loss_fraction';
            const lossKey = isCumulative ? 'cumulative_loss' : 'value_loss';

            if (colorMode === 'gdp_pct' && !isCumulative && window.gdpImpactLookup) {
                // Use loss_as_gdp_pct from gdpImpactData where available (annual scenarios only)
                colorValues = filtered.map(c => {
                    if (valueType === 'all') return c.loss_as_gdp_pct || 0;
                    const rec = window.gdpImpactLookup[`${scenario}||${model}||${valueType}||${c.country}`];
                    return rec ? rec.loss_as_gdp_pct : 0;
                });
                colorbarTitle = '% GDP';
                colorbarFormat = '.1f';
            } else if (colorMode === 'value_loss') {
                // Color by absolute value loss
                colorValues = filtered.map(c => (c[lossKey] || 0) / 1e6); // Convert to millions for color scale
                colorbarTitle = isCumulative ? 'Cumulative Loss ($M)' : 'Value Loss ($M)';
                colorbarFormat = ',.0f';
            } else {
                // Default: colour by reef tourism loss fraction
                colorValues = filtered.map(c => c[reefFractionKey] || 0);
                colorbarTitle = isCumulative ? 'Cumulative Loss %' : 'Loss %';
                colorbarFormat = '.0%';
            }
            
            const trace = {
                y: filtered.map(c => c.country),
                x: xData,
                type: 'bar',
                orientation: 'h',
                marker: {
                    color: colorValues,
                    colorscale: [[0, '#22c55e'], [0.25, '#eab308'], [0.5, '#E3B710'], [1, '#F11B00']],
                    colorbar: {
                        title: colorbarTitle,
                        tickformat: colorbarFormat
                    }
                },
                text: filtered.map(textFn),
                textposition: 'outside',
                hovertemplate:
                    'Country: %{y}<br>' +
                    `${xTitle}: %{x:.2f}<extra></extra>`
            };
            
            const layout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                xaxis: { 
                    gridcolor: '#334155',
                    title: xTitle
                },
                yaxis: { 
                    autorange: 'reversed'
                },
                margin: { t: 20, r: 100, b: 60, l: 150 }
            };
            
            Plotly.newPlot('country-chart', [trace], layout, {responsive: true});
        }
        
        function renderGdpChart() {
            if (!gdpImpactData) return;
            
            const scenario = document.getElementById('gdp-scenario').value;
            const model = document.getElementById('gdp-model').value;
            const limitValue = document.getElementById('gdp-limit').value;
            const metric = document.getElementById('gdp-metric').value;
            
            // Match scenario exactly
            let filtered = gdpImpactData.filter(c => 
                c.scenario === scenario && c.model === model
            );
            
            // Filter and sort by selected metric
            if (metric === 'loss_as_gdp_pct') {
                filtered = filtered.filter(c => c.loss_as_gdp_pct > 0)
                    .sort((a, b) => b.loss_as_gdp_pct - a.loss_as_gdp_pct);
            } else {
                filtered = filtered.filter(c => c.value_loss > 0)
                    .sort((a, b) => b.value_loss - a.value_loss);
            }
            
            // Apply limit
            if (limitValue !== 'all') {
                filtered = filtered.slice(0, parseInt(limitValue));
            }
            
            // Dynamic height based on number of countries
            const chartHeight = Math.max(500, filtered.length * 30);
            document.getElementById('gdp-chart').style.height = chartHeight + 'px';
            
            // Configure based on metric
            let xData, xTitle, textFn, colorbarTitle, colorbarFormat;
            if (metric === 'loss_as_gdp_pct') {
                xData = filtered.map(c => c.loss_as_gdp_pct);
                xTitle = 'Projected Loss as % of National GDP';
                textFn = c => `${c.loss_as_gdp_pct.toFixed(2)}%`;
                colorbarTitle = '% GDP';
                colorbarFormat = '.1f';
            } else {
                xData = filtered.map(c => c.value_loss / 1e6);
                xTitle = 'Value Loss ($ Million)';
                textFn = c => `$${(c.value_loss / 1e6).toFixed(1)}M`;
                colorbarTitle = 'Loss $M';
                colorbarFormat = '.0f';
            }
            
            const trace = {
                y: filtered.map(c => c.country),
                x: xData,
                type: 'bar',
                orientation: 'h',
                marker: {
                    color: filtered.map(c => c.loss_as_gdp_pct),
                    colorscale: [[0, '#22c55e'], [0.1, '#eab308'], [0.3, '#E3B710'], [1, '#F11B00']],
                    colorbar: {
                        title: colorbarTitle,
                        tickformat: colorbarFormat
                    }
                },
                text: filtered.map(textFn),
                textposition: 'outside',
                hovertemplate:
                    'Country: %{y}<br>' +
                    `${xTitle}: %{x:.2f}<extra></extra>`
            };
            
            const layout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                xaxis: { 
                    gridcolor: '#334155',
                    title: xTitle
                },
                yaxis: { 
                    autorange: 'reversed'
                },
                margin: { t: 20, r: 100, b: 0, l: 150 }
            };
            
            Plotly.newPlot('gdp-chart', [trace], layout, {responsive: true});
        }
        
        function renderGdpComparison() {
            if (!gdpImpactData) return;
            
            const model = document.getElementById('gdp-comparison-model').value;
            const metric = document.getElementById('gdp-comparison-metric').value;
            const limitValue = document.getElementById('gdp-comparison-limit').value;
            const valueType = getSelectedValueType('gdp-value-type', 'all');
            const limit = limitValue === 'all' ? 9999 : parseInt(limitValue);
            let gdpSource = gdpImpactData.filter(c => c.model === model);
            if (valueType !== 'all') {
                gdpSource = gdpSource.filter(c => c.value_type === valueType);
            } else {
                const grouped = new Map();
                gdpSource.forEach((row) => {
                    const key = `${row.scenario}||${row.model}||${row.country}`;
                    if (!grouped.has(key)) {
                        grouped.set(key, {
                            ...row,
                            value_type: 'all',
                            value_loss: 0,
                            loss_as_gdp_pct: 0,
                        });
                    }
                    const g = grouped.get(key);
                    g.value_loss += Number(row.value_loss || 0);
                    const nationalGdp = Number(row.national_gdp || 0);
                    if (nationalGdp > 0) {
                        g.loss_as_gdp_pct = (g.value_loss / nationalGdp) * 100;
                    }
                });
                gdpSource = Array.from(grouped.values());
            }
            
            // Get top countries by worst-case impact for selected model and metric
            const worstCase = gdpSource.filter(c => 
                c.scenario.includes('rcp85') && c.scenario.includes('2100') &&
                c.model === model
            );
            
            // Sort by selected metric
            const metricKey = metric === 'loss_as_gdp_pct' ? 'loss_as_gdp_pct' : 'value_loss';
            worstCase.sort((a, b) => (b[metricKey] || 0) - (a[metricKey] || 0));
            const topCountries = worstCase.slice(0, limit).map(c => c.country);
            
            // Dynamic height based on number of countries - increased spacing
            const chartHeight = Math.max(1000, topCountries.length * 100);
            document.getElementById('gdp-comparison-chart').style.height = chartHeight + 'px';
            
            // Create traces: 2050 overlaid on 2100 for each RCP (2 bars per country)
            // 2100 is the full bar (lighter), 2050 is overlaid on top (darker)
            // Use custom y positions to group RCP 4.5 and RCP 8.5 side by side
            const rcpScenarios = [
                { rcp: 'rcp45', color: '#3A9AB2', name: 'RCP 4.5', yOffset: -0.2 },
                { rcp: 'rcp85', color: '#F11B00', name: 'RCP 8.5', yOffset: 0.2 },
            ];
            
            const traces = [];
            
            // Create custom y positions for each country to offset RCP bars
            // Use larger spacing (multiply by 1.5) to spread countries out more
            const createYPositions = (offset) => {
                return topCountries.map((country, idx) => {
                    // Use numeric index with larger spacing and offset to create side-by-side bars
                    return idx * 1.5 + offset;
                });
            };
            
            // For each RCP, create two traces: 2100 (base, lighter) and 2050 (overlay, darker)
            rcpScenarios.forEach(rcp => {
                // Get data for both years
                const data2100 = gdpSource.filter(c => 
                    c.scenario.toLowerCase().includes(rcp.rcp) && 
                    c.scenario.includes('2100') &&
                    c.model === model &&
                    topCountries.includes(c.country)
                );
                
                const data2050 = gdpSource.filter(c => 
                    c.scenario.toLowerCase().includes(rcp.rcp) && 
                    c.scenario.includes('2050') &&
                    c.model === model &&
                    topCountries.includes(c.country)
                );
                
                const countryMap2100 = {};
                data2100.forEach(d => { countryMap2100[d.country] = d[metricKey] || 0; });
                
                const countryMap2050 = {};
                data2050.forEach(d => { countryMap2050[d.country] = d[metricKey] || 0; });
                
                const yPositions = createYPositions(rcp.yOffset);
                
                // 2100 trace (base, lighter opacity) - full height
                traces.push({
                    y: yPositions,
                    x: topCountries.map(c => countryMap2100[c] || 0),
                    customdata: topCountries,
                    name: `${rcp.name} - 2100`,
                    type: 'bar',
                    orientation: 'h',
                    hovertemplate:
                        'Country: %{customdata}<br>' +
                        '%{fullData.name}<br>' +
                        `${metric === 'loss_as_gdp_pct' ? 'Projected Loss as GDP: %{x:.2f}%' : 'Projected Value Loss: $%{x:,.0f}'}` +
                        '<extra></extra>',
                    width: 0.4,  // Narrower bars
                    marker: {
                        color: rcp.color,
                        opacity: 0.5,  // Lighter for base
                        line: { width: 0 }
                    }
                });
                
                // 2050 trace (overlay, darker opacity) - overlaid on top of 2100
                // Since 2050 < 2100, this will show as a darker section on the lighter bar
                traces.push({
                    y: yPositions,
                    x: topCountries.map(c => countryMap2050[c] || 0),
                    customdata: topCountries,
                    name: `${rcp.name} - 2050`,
                    type: 'bar',
                    orientation: 'h',
                    hovertemplate:
                        'Country: %{customdata}<br>' +
                        '%{fullData.name}<br>' +
                        `${metric === 'loss_as_gdp_pct' ? 'Projected Loss as GDP: %{x:.2f}%' : 'Projected Value Loss: $%{x:,.0f}'}` +
                        '<extra></extra>',
                    width: 0.4,  // Narrower bars
                    marker: {
                        color: rcp.color,
                        opacity: 1.0,  // Darker for overlay
                        line: { width: 0 }
                    }
                });
            });
            
            // Determine axis title based on metric
            const xAxisTitle = metric === 'loss_as_gdp_pct' 
                ? 'Projected Loss as % of National GDP'
                : 'Projected Value Loss ($ Million)';
            
            // Format function for text labels
            const formatValue = (val) => {
                if (metric === 'loss_as_gdp_pct') {
                    return `${val.toFixed(2)}%`;
                } else {
                    if (val >= 1e6) return `$${(val / 1e6).toFixed(1)}M`;
                    if (val >= 1e3) return `$${(val / 1e3).toFixed(1)}k`;
                    return `$${val.toFixed(0)}`;
                }
            };
            
            // Create custom y-axis tick labels (country names at their base positions)
            // Match the spacing used in createYPositions
            const yTickPositions = topCountries.map((_, idx) => idx * 1.5);
            const yTickLabels = topCountries;
            
            const layout = {
                barmode: 'overlay',  // Overlay 2050 on 2100 within each RCP
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                xaxis: { 
                    gridcolor: '#334155',
                    title: xAxisTitle
                },
                yaxis: { 
                    tickmode: 'array',
                    tickvals: yTickPositions,
                    ticktext: yTickLabels,
                    autorange: 'reversed'
                },
                legend: { 
                    orientation: 'h', 
                    y: -0.15,
                    font: { size: 11 }
                },
                margin: { t: 20, r: 20, b: 80, l: 150 }
            };
            
            Plotly.newPlot('gdp-comparison-chart', traces, layout, {responsive: true});
        }
        
        function renderModelComparison() {
            if (!modelCurves) return;
            
            // Standard models (linear and compound)
            const standardTraces = Object.entries(modelCurves)
                .filter(([key]) => key === 'linear' || key === 'compound')
                .map(([key, curve]) => ({
                    x: curve.delta_cc,
                    y: curve.remaining_value,
                    name: curve.name,
                    hovertemplate:
                        `${curve.name}<br>` +
                        'Change in Coral Cover: %{x:.1f}pp<br>' +
                        'Remaining Value: %{y:.1f}%<extra></extra>',
                    mode: 'lines',
                    line: {
                        color: key === 'linear' ? '#3A9AB2' : '#F11B00',
                        width: 3
                    }
                }));
            
            const standardLayout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                autosize: true,
                xaxis: { 
                    gridcolor: '#334155',
                    title: 'Change in Coral Cover (percentage points)',
                    zeroline: true,
                    zerolinecolor: '#64748b',
                    range: [-50, 0]
                },
                yaxis: { 
                    gridcolor: '#334155',
                    title: 'Remaining Value (%)',
                    range: [0, 100]
                },
                legend: { 
                    orientation: 'h', 
                    y: -0.15
                },
                margin: { t: 20, r: 30, b: 70, l: 60 },
                shapes: [{
                    type: 'line',
                    x0: 0, x1: 0,
                    y0: 0, y1: 120,
                    line: { color: '#64748b', width: 1, dash: 'dot' }
                }]
            };
            
            Plotly.newPlot('model-chart', standardTraces, standardLayout, {responsive: true});
            
            // Tipping point model with multiple original_cc values
            const tippingPointTraces = Object.entries(modelCurves)
                .filter(([key]) => key.startsWith('tipping_point_'))
                .sort(([key1], [key2]) => {
                    const cc1 = modelCurves[key1].original_cc || 0;
                    const cc2 = modelCurves[key2].original_cc || 0;
                    return cc1 - cc2;
                })
                .map(([key, curve]) => {
                    const ogCc = curve.original_cc || 0.5;
                    // Get color from reds colormap (darker for lower initial cover)
                    // Limit redIntensity to avoid very dark colors (max 0.7 instead of 1.0)
                    const redIntensity = Math.max(0.3, Math.min(0.7, 0.3 + (ogCc - 0.1) * 0.4 / 0.6));
                    const color = `rgb(${Math.round(220 * (1 - redIntensity))}, ${Math.round(38 * (1 - redIntensity))}, ${Math.round(38 * (1 - redIntensity))})`;
                    
                    return {
                        x: curve.delta_cc,
                        y: curve.remaining_value,
                        name: curve.name,
                        hovertemplate:
                            `${curve.name}<br>` +
                            'Change in Coral Cover: %{x:.1f}pp<br>' +
                            'Remaining Value: %{y:.1f}%<extra></extra>',
                        mode: 'lines',
                        line: {
                            color: color,
                            width: 3
                        }
                    };
                });
            
            // Add markers at tipping points
            const tippingPointMarkers = Object.entries(modelCurves)
                .filter(([key]) => key.startsWith('tipping_point_'))
                .map(([key, curve]) => {
                    const ogCc = curve.original_cc || 0.5;
                    // Find where value drops to near zero (tipping point)
                    const zeroIdx = curve.remaining_value.findIndex((v, i) => 
                        i > 0 && v < 1 && curve.remaining_value[i - 1] > 1
                    );
                    if (zeroIdx === -1) return null;
                    
                    const redIntensity = Math.max(0.3, Math.min(1, 0.3 + (ogCc - 0.1) * 0.7 / 0.6));
                    const color = `rgb(${Math.round(220 * (1 - redIntensity))}, ${Math.round(38 * (1 - redIntensity))}, ${Math.round(38 * (1 - redIntensity))})`;
                    
                    return {
                        x: [curve.delta_cc[zeroIdx]],
                        y: [curve.remaining_value[zeroIdx]],
                        name: `${curve.name} (tipping point)`,
                        hovertemplate:
                            `${curve.name} (tipping point)<br>` +
                            'Change in Coral Cover: %{x:.1f}pp<br>' +
                            'Remaining Value: %{y:.1f}%<extra></extra>',
                        mode: 'markers',
                        marker: {
                            symbol: 'x',
                            size: 12,
                            color: 'black',
                            line: { width: 1, color: 'black' }
                        },
                        showlegend: false
                    };
                })
                .filter(t => t !== null);
            
            const tippingPointLayout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Instrument Sans' },
                hoverlabel: { namelength: -1 },
                autosize: true,
                title: {
                    text: 'Tipping Point Model: Effect of Initial Coral Cover',
                    font: { size: 14, color: '#e2e8f0' },
                    x: 0.5
                },
                xaxis: { 
                    gridcolor: '#334155',
                    title: 'Change in Coral Cover (percentage points)',
                    zeroline: true,
                    zerolinecolor: '#64748b',
                    range: [-50, 0]
                },
                yaxis: { 
                    gridcolor: '#334155',
                    title: 'Remaining Value (%)',
                    range: [0, 100]
                },
                legend: { 
                    orientation: 'h', 
                    y: -0.15
                },
                margin: { t: 50, r: 30, b: 70, l: 60 },
                shapes: [{
                    type: 'line',
                    x0: 0, x1: 0,
                    y0: 0, y1: 120,
                    line: { color: '#64748b', width: 1, dash: 'dot' }
                }]
            };
            
            Plotly.newPlot('tipping-point-chart', [...tippingPointTraces, ...tippingPointMarkers], tippingPointLayout, {responsive: true});
        }
        
        // ============================================================
        // MAP FUNCTIONS
        // ============================================================
        
        function initializeMap() {
            // Use canvas renderer for better performance with many polygons
            map = L.map('map', {
                preferCanvas: true,  // Use canvas renderer instead of SVG for better performance
                zoomControl: true
            }).setView([0, 0], 2);
            
            L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; OpenStreetMap, &copy; CARTO',
                maxZoom: 18
            }).addTo(map);

            // Pane ordering (Leaflet tilePane = 200):
            //   siteBackPane    250 – fisheries/coastal grid cells
            //   choroplethPane  380 – country choropleth
            //   sitePointPane   500 – legacy point markers
            //   sitePolygonPane 700 – tourism reef polygons (always top geometry)
            map.createPane('siteBackPane');
            map.getPane('siteBackPane').style.zIndex = 250;
            map.createPane('choroplethPane');
            map.getPane('choroplethPane').style.zIndex = 380;
            map.createPane('sitePointPane');
            map.getPane('sitePointPane').style.zIndex = 500;
            map.createPane('sitePolygonPane');
            map.getPane('sitePolygonPane').style.zIndex = 700;

            // Dedicated renderers pinned to their panes so canvas batching cannot
            // paint grid cells above tourism polygons.
            siteGridRenderer = L.canvas({ pane: 'siteBackPane', padding: 0.5 });
            siteTourismRenderer = L.svg({ pane: 'sitePolygonPane' });
            
            choroplethLayer = L.layerGroup();
            siteGridLayer = L.layerGroup().addTo(map);
            siteTourismLayer = L.layerGroup().addTo(map);
            siteLayer = L.layerGroup().addTo(map);
            
            // Initialize legend with default settings (annual loss %, non-cumulative)
            const defaultScale = getColorScale(false, 'loss_percent');
            updateMapLegend(defaultScale, false);
            
            // Re-render on zoom/move for viewport-based rendering
            let zoomMoveTimeout = null;
            map.on('zoomend moveend', () => {
                if (!isRendering) {
                    clearTimeout(zoomMoveTimeout);
                    zoomMoveTimeout = setTimeout(() => {
                        // Reload chunked GeoJSON on viewport changes.
                        // In vector tile mode, tile loading is handled natively by Leaflet.VectorGrid.
                        if (!useVectorTilesForSites) {
                            loadSiteData();
                        }
                    }, 300);  // Debounce zoom/move events
                }
            });
            
            // Load country boundaries (using Natural Earth via CDN)
            loadCountryBoundaries();
        }
        
        async function loadCountryBoundaries() {
            try {
                // Prefer high-resolution country boundaries.
                const boundarySources = [
                    'https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson',
                    'https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson',
                ];
                for (const url of boundarySources) {
                    try {
                        const response = await fetch(url);
                        if (response.ok) {
                            countryBoundaries = await response.json();
                            console.log(`Loaded country boundaries from: ${url}`);
                            return;
                        }
                    } catch (err) {
                        console.warn(`Boundary fetch failed for ${url}:`, err);
                    }
                }
                console.warn('Could not load country boundaries, choropleth will be unavailable');
            } catch (error) {
                console.warn('Error loading country boundaries:', error);
            }
        }

        function clearSiteDataLayers() {
            if (siteGridLayer) siteGridLayer.clearLayers();
            if (siteTourismLayer) siteTourismLayer.clearLayers();
            if (siteLayer) siteLayer.clearLayers();
        }

        function bringTourismLayersToFront() {
            if (siteTourismLayer) {
                siteTourismLayer.bringToFront();
            }
        }
        
        async function loadSiteData() {
            // Wait for siteManifest to be loaded
            if (!siteManifest) {
                console.warn('siteManifest not loaded yet, waiting...');
                // Try again after a short delay
                setTimeout(loadSiteData, 500);
                return;
            }
            
            if (!map || !siteLayer) {
                console.warn('Map not initialized yet');
                return;
            }
            
            // Store current view before loading new data (if not initial load)
            let currentView = null;
            if (!isInitialMapLoad && map) {
                currentView = {
                    center: map.getCenter(),
                    zoom: map.getZoom()
                };
            }
            
            const scenario = document.getElementById('map-scenario').value;
            const model = document.getElementById('map-model').value;
            const selectedValueTypes = getMapSelectedValueTypes();
            
            if (!scenario || !model) {
                console.warn('Scenario or model not selected');
                return;
            }
            
            // Check if this is a cumulative scenario
            const isCumulative = scenario.startsWith('cumulative_');
            
            // For cumulative, scenario is like "cumulative_rcp45_2050", need to extract rcp and year
            let scenarioKey = scenario;
            if (isCumulative) {
                // Extract rcp and year from "cumulative_rcp45_2050"
                const parts = scenario.replace('cumulative_', '').split('_');
                const rcp = parts[0]; // "rcp45"
                const year = parts[1]; // "2050"
                scenarioKey = `cumulative_${rcp}_${year}`;
            }
            
            // Sanitize model name to match export format
            const sanitizeModelName = (name) => {
                return name.replace(/\s+/g, '_')
                    .replace(/\//g, '_')
                    .replace(/%/g, 'pct')
                    .replace(/[()]/g, '');
            };
            
            const sanitizedModel = sanitizeModelName(model);
            const metric = document.getElementById('map-metric').value;
            const scale = getColorScale(isCumulative, metric);
            const showChoropleth = document.getElementById('map-choropleth-toggle').checked;
            if (selectedValueTypes.length === 0) {
                clearSiteDataLayers();
                currentSiteGeojson = null;
                setMapEmptyState('Select at least one dataset to view map values.');
                updateMapLegend(scale, showChoropleth, null, 'No datasets selected');
                if (choroplethLayer) {
                    map.removeLayer(choroplethLayer);
                    choroplethLayer = L.layerGroup();
                }
                removeChoroplethLegend();
                return;
            }
            setMapEmptyState('');
            const buildScenarioKey = (datasetKey) => `${datasetKey}_${scenarioKey}_${sanitizedModel}`;
            const datasetsToLoad = selectedValueTypes;
            const scenarioDatasetKeys = datasetsToLoad.map(buildScenarioKey);

            // Prefer compact gridded export (sites_grid_*.json + sites_metrics_*.json).
            const griddedManifest = isCumulative
                ? (siteManifest?.gridded_sites_cumulative || {})
                : (siteManifest?.gridded_sites_annual || {});
            const canUseGridded = datasetsToLoad.every((k) => Boolean(griddedManifest[k]));

            if (canUseGridded) {
                try {
                    console.log('Loading gridded site data', {
                        scenario,
                        model,
                        scenarioDatasetKeys,
                        datasetsToLoad,
                    });
                    const featureLists = await Promise.all(
                        datasetsToLoad.map((datasetKey) =>
                            loadGriddedSiteFeatures({
                                datasetKey,
                                scenarioDatasetKey: buildScenarioKey(datasetKey),
                                griddedEntry: griddedManifest[datasetKey],
                            })
                        )
                    );
                    const features = featureLists.flat();
                    if (features.length === 0) {
                        setMapEmptyState('No map data found for selected datasets.');
                        updateMapLegend(scale, showChoropleth, null, 'No gridded data for selection');
                        clearSiteDataLayers();
                        currentSiteGeojson = null;
                        return;
                    }
                    setMapEmptyState('');
                    console.log(`✓ Loaded ${features.length} gridded cell features`);
                    renderSites({ type: 'FeatureCollection', features }, isCumulative, currentView);
                    isInitialMapLoad = false;
                    return;
                } catch (error) {
                    console.error('Error loading gridded site data:', error);
                    setMapEmptyState('Error loading gridded map data.');
                    return;
                }
            }

            const vectorTileIndex = siteManifest?.vector_tile_scenarios || {};
            const pointChunkIndex = siteManifest?.site_point_chunks || {};
            const chunkZoom = Number(siteManifest?.site_point_chunk_zoom);
            const datasetTileIndex = isCumulative
                ? (siteManifest?.site_dataset_tiles_cumulative || {})
                : (siteManifest?.site_dataset_tiles_annual || {});
            const datasetTileZoom = Number(
                isCumulative
                    ? siteManifest?.site_dataset_tile_zoom_cumulative
                    : siteManifest?.site_dataset_tile_zoom_annual
            );
            const effectiveTileZoom = Number.isFinite(datasetTileZoom) ? datasetTileZoom : chunkZoom;
            const visibleTileKeys = Number.isFinite(effectiveTileZoom)
                ? getTileKeysForBounds(map.getBounds(), effectiveTileZoom)
                : [];
            const minPointMapZoom = Number(siteManifest?.site_dataset_min_map_zoom ?? 4);
            const canLoadDatasetWidePointsAtZoom = map.getZoom() >= minPointMapZoom;
            const canUseDatasetWideTiles = Boolean(
                scenarioDatasetKeys.length > 0 &&
                canLoadDatasetWidePointsAtZoom &&
                datasetsToLoad.every((datasetKey) => {
                    const entry = datasetTileIndex?.[datasetKey];
                    return entry && entry.geometry && entry.attributes;
                })
            );

            const canUseVectorTiles = Boolean(
                L?.vectorGrid &&
                scenarioDatasetKeys.length > 0 &&
                !canUseDatasetWideTiles &&
                scenarioDatasetKeys.every((k) => Boolean(vectorTileIndex[k]?.url_template))
            );

            if (canUseVectorTiles) {
                useVectorTilesForSites = true;
                console.log('Loading site data using vector tiles', {
                    scenario,
                    model,
                    selectedValueTypes,
                    scenarioDatasetKeys,
                });
                renderVectorTileSites(
                    scenarioDatasetKeys,
                    vectorTileIndex,
                    isCumulative,
                    currentView
                );
                isInitialMapLoad = false;
                return;
            }

            useVectorTilesForSites = false;

            const filesToFetch = new Set();
            scenarioDatasetKeys.forEach((scenarioDatasetKey) => {
                filesToFetch.add(`sites_${scenarioDatasetKey}.json`);
                if (canUseDatasetWideTiles) {
                    return;
                }
                const tileMap = pointChunkIndex[scenarioDatasetKey];
                if (tileMap && visibleTileKeys.length > 0) {
                    visibleTileKeys.forEach((tileKey) => {
                        const chunkFile = tileMap[tileKey];
                        if (chunkFile) {
                            filesToFetch.add(chunkFile);
                        }
                    });
                }
            });
            const filenames = Array.from(filesToFetch);
            
            console.log('Loading site data:', {
                scenario,
                model,
                selectedValueTypes,
                isCumulative,
                scenarioKey,
                sanitizedModel,
                chunkZoom: effectiveTileZoom,
                mapZoom: map.getZoom(),
                minPointMapZoom,
                canUseDatasetWideTiles,
                requestedTiles: visibleTileKeys.length,
                filenames,
            });
            
            try {
                const [baseGeojsons, datasetWidePointFeatures] = await Promise.all([
                    Promise.all(
                        scenarioDatasetKeys.map((scenarioDatasetKey) =>
                            fetchGeoJsonWithCache(`sites_${scenarioDatasetKey}.json`)
                        )
                    ),
                    canUseDatasetWideTiles
                        ? Promise.all(
                            datasetsToLoad.map((datasetKey) =>
                                loadDatasetWidePointTileFeatures({
                                    datasetKey,
                                    scenarioDatasetKey: buildScenarioKey(datasetKey),
                                    tileKeys: visibleTileKeys,
                                    datasetTileIndex,
                                })
                            )
                        )
                        : Promise.resolve([]),
                ]);
                const validBaseGeojsons = baseGeojsons.filter(g => g && Array.isArray(g.features));
                let features = validBaseGeojsons.flatMap((g) => g.features || []);

                if (canUseDatasetWideTiles) {
                    features = features.concat(datasetWidePointFeatures.flat());
                } else {
                    const extraGeojsons = await Promise.all(
                        filenames
                            .filter((name) => !name.startsWith('sites_'))
                            .map((filename) => fetchGeoJsonWithCache(filename))
                    );
                    const validExtraGeojsons = extraGeojsons.filter(g => g && Array.isArray(g.features));
                    features = features.concat(validExtraGeojsons.flatMap((g) => g.features || []));
                }

                if (features.length === 0) {
                    console.error('No site datasets found for selection', { filenames });
                    clearSiteDataLayers();
                    currentSiteGeojson = null;
                    if (!canLoadDatasetWidePointsAtZoom && datasetsToLoad.some((k) => Boolean(datasetTileIndex?.[k]))) {
                        setMapEmptyState(`Zoom in to at least ${minPointMapZoom} to load point datasets.`);
                        updateMapLegend(scale, showChoropleth, null, 'Point datasets load when zoomed in');
                    } else {
                        setMapEmptyState('No map data found for selected datasets.');
                        updateMapLegend(scale, showChoropleth, null, 'No datasets with available map data');
                    }
                    return;
                }
                setMapEmptyState('');
                const geojson = {
                    type: 'FeatureCollection',
                    features,
                };
                
                if (!geojson || !geojson.features || geojson.features.length === 0) {
                    console.warn('GeoJSON has no features for current viewport');
                    clearSiteDataLayers();
                    currentSiteGeojson = null;
                    setMapEmptyState('No features available in the current map view.');
                    updateMapLegend(scale, showChoropleth, null, 'No features in current viewport');
                    return;
                }
                
                console.log(`✓ Loaded ${geojson.features.length} features from ${filenames.length} file(s)`);
                renderSites(geojson, isCumulative, currentView);
                
                // Mark that initial load is complete
                isInitialMapLoad = false;
            } catch (error) {
                console.error('Error loading site data:', error);
                console.error('Error details:', error.message, error.stack);
                clearSiteDataLayers();
                currentSiteGeojson = null;
            }
        }
        
        // Color scales for different metrics
        const COLOR_SCALES = {
            // Green to yellow to red scale
            loss_percent_annual: {
                breaks: [0.1, 0.25, 0.5],
                colors: ['#22c55e', '#eab308', '#E3B710', '#F11B00'],
                labels: ['< 10%', '10–25%', '25–50%', '> 50%'],
                title: 'Annual Loss %'
            },
            loss_percent_cumulative: {
                // For cumulative: use annual loss % which is in loss_fraction
                breaks: [0.1, 0.25, 0.5],
                colors: ['#22c55e', '#eab308', '#E3B710', '#F11B00'],
                labels: ['< 10%', '10–25%', '25–50%', '> 50%'],
                title: 'Annual Loss % (at endpoint)'
            },
            absolute_annual: {
                breaks: [10000, 100000, 1000000],
                colors: ['#22c55e', '#eab308', '#E3B710', '#F11B00'],
                labels: ['< $10k', '$10k–$100k', '$100k–$1000k', '> $1000k'],
                title: 'Annual Loss ($)'
            },
            absolute_cumulative: {
                breaks: [10000, 100000, 1000000],
                colors: ['#22c55e', '#eab308', '#E3B710', '#F11B00'],
                labels: ['< $10k', '$10k–$100k', '$100k–$1000k', '> $1000k'],
                title: 'Cumulative Loss ($)'
            }
        };
        
        function getColorScale(isCumulative, metric) {
            if (metric === 'loss_percent') {
                return isCumulative ? COLOR_SCALES.loss_percent_cumulative : COLOR_SCALES.loss_percent_annual;
            } else {
                return isCumulative ? COLOR_SCALES.absolute_cumulative : COLOR_SCALES.absolute_annual;
            }
        }
        
        function getColorFromScale(value, scale) {
            if (value < scale.breaks[0]) return scale.colors[0];
            if (value < scale.breaks[1]) return scale.colors[1];
            if (value < scale.breaks[2]) return scale.colors[2];
            return scale.colors[3];
        }
        
        function formatLegendValue(value, scaleTitle = '') {
            if (value == null || !Number.isFinite(value)) return 'n/a';
            const title = (scaleTitle || '').toLowerCase();
            if (title.includes('%')) {
                return `${(value * 100).toFixed(1)}%`;
            }
            if (Math.abs(value) >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
            if (Math.abs(value) >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
            if (Math.abs(value) >= 1e3) return `$${(value / 1e3).toFixed(1)}k`;
            return `$${Number(value).toFixed(0)}`;
        }

        function setMapEmptyState(message = '') {
            const el = document.getElementById('map-empty-state');
            if (!el) return;
            if (message) {
                el.textContent = message;
                el.style.display = 'flex';
            } else {
                el.style.display = 'none';
            }
        }

        function updateMapLegend(scale, showChoropleth = false, pointRange = null, statusMessage = '') {
            const legendEl = document.getElementById('map-legend');
            if (!legendEl) return;
            
            let html = `<div class="legend-title" style="font-size: 0.8rem; color: var(--text-secondary); margin-right: 1rem;">${scale.title}</div>`;
            
            for (let i = 0; i < scale.colors.length; i++) {
                html += `
                    <div class="legend-item">
                        <div class="legend-color" style="background: ${scale.colors[i]};"></div>
                        <span>${scale.labels[i]}</span>
                    </div>`;
            }
            
            if (showChoropleth) {
                html += `
                    <div class="legend-separator" style="border-left: 1px solid var(--border); height: 24px; margin: 0 1rem;"></div>
                    <div class="legend-item" id="choropleth-legend">
                        <span style="color: var(--text-secondary); font-size: 0.85rem;">Choropleth: Country-level aggregates</span>
                    </div>`;
            }

            if (pointRange && Number.isFinite(pointRange.min) && Number.isFinite(pointRange.max)) {
                html += `
                    <div class="legend-colorbar">
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">Point colourbar</div>
                        <div class="legend-colorbar-gradient"></div>
                        <div class="legend-colorbar-labels">
                            <span>${formatLegendValue(pointRange.min, scale.title)}</span>
                            <span>${formatLegendValue(pointRange.max, scale.title)}</span>
                        </div>
                    </div>`;
            }

            if (statusMessage) {
                html += `
                    <div class="legend-item">
                        <span style="color: var(--text-secondary); font-size: 0.85rem;">${statusMessage}</span>
                    </div>`;
            }
            
            legendEl.innerHTML = html;
        }
        
        // Simple geometry simplification function (only at very low zoom levels)
        function simplifyGeometry(geometry, tolerance = 0.0001) {
            // Only simplify if tolerance is very high (low zoom), otherwise return original
            if (tolerance < 0.0005) {
                return geometry;  // Don't simplify at normal zoom levels
            }
            
            if (!geometry || !geometry.coordinates) return geometry;
            
            function simplifyRing(ring, tol) {
                if (ring.length <= 2) return ring;
                const simplified = [ring[0]];
                for (let i = 1; i < ring.length - 1; i++) {
                    const prev = ring[i - 1];
                    const curr = ring[i];
                    const next = ring[i + 1];
                    // Simple distance check - keep point if it's far enough from line
                    const dx1 = curr[0] - prev[0];
                    const dy1 = curr[1] - prev[1];
                    const dx2 = next[0] - prev[0];
                    const dy2 = next[1] - prev[1];
                    const cross = Math.abs(dx1 * dy2 - dx2 * dy1);
                    if (cross > tol) {
                        simplified.push(curr);
                    }
                }
                simplified.push(ring[ring.length - 1]);
                return simplified;
            }
            
            if (geometry.type === 'Polygon') {
                return {
                    type: 'Polygon',
                    coordinates: geometry.coordinates.map(ring => simplifyRing(ring, tolerance))
                };
            } else if (geometry.type === 'MultiPolygon') {
                return {
                    type: 'MultiPolygon',
                    coordinates: geometry.coordinates.map(polygon => 
                        polygon.map(ring => simplifyRing(ring, tolerance))
                    )
                };
            }
            return geometry;
        }
        
        // Extra cells kept around the viewport so partially visible grid cells / polygons
        // are not dropped when panning or zooming.
        const VIEWPORT_CELL_BUFFER = 2;
        const GRID_CELL_SIZE_DEG = 0.5;

        // Filter features by viewport bounds
        function filterFeaturesByViewport(geojson, mapBounds) {
            if (!mapBounds) return geojson;

            const pad = VIEWPORT_CELL_BUFFER * GRID_CELL_SIZE_DEG;
            const west = mapBounds.getWest() - pad;
            const east = mapBounds.getEast() + pad;
            const south = mapBounds.getSouth() - pad;
            const north = mapBounds.getNorth() + pad;
            
            const filteredFeatures = geojson.features.filter(feature => {
                if (!feature.geometry || !feature.geometry.coordinates) return false;
                
                // Get bounding box of feature
                let minLon = Infinity, maxLon = -Infinity;
                let minLat = Infinity, maxLat = -Infinity;
                
                function processCoords(coords) {
                    if (Array.isArray(coords[0])) {
                        coords.forEach(processCoords);
                    } else {
                        const [lon, lat] = coords;
                        minLon = Math.min(minLon, lon);
                        maxLon = Math.max(maxLon, lon);
                        minLat = Math.min(minLat, lat);
                        maxLat = Math.max(maxLat, lat);
                    }
                }
                
                processCoords(feature.geometry.coordinates);
                
                return !(maxLon < west || minLon > east || maxLat < south || minLat > north);
            });
            
            return {
                type: 'FeatureCollection',
                features: filteredFeatures
            };
        }

        function renderVectorTileSites(scenarioDatasetKeys, vectorTileIndex, isCumulative = false, preserveView = null) {
            if (!map || !siteLayer) {
                return;
            }
            if (isRendering) {
                return;
            }
            isRendering = true;
            clearSiteDataLayers();
            currentSiteGeojson = null;

            const metric = document.getElementById('map-metric').value;
            const scale = getColorScale(isCumulative, metric);
            const showChoropleth = document.getElementById('map-choropleth-toggle').checked;
            const zoom = map.getZoom();
            const lossType = isCumulative ? 'Cumulative' : 'Annual';

            const formatMoney = (val) => {
                if (val >= 1e6) return `$${(val / 1e6).toFixed(2)}M`;
                if (val >= 1e3) return `$${(val / 1e3).toFixed(1)}k`;
                return `$${val.toFixed(0)}`;
            };

            const getFeatureColorValue = (props) => {
                const lossFraction = Number(props?.loss_fraction || 0);
                const lossValue = isCumulative
                    ? Number(props?.cumulative_loss || 0)
                    : Number(props?.value_loss || props?.annual_loss || 0);
                return metric === 'loss_percent' ? lossFraction : lossValue;
            };

            const styleByGeometry = (properties, z, geometryDimension) => {
                const colorValue = getFeatureColorValue(properties || {});
                const fillColor = getColorFromScale(colorValue, scale);
                // Use a geometry-agnostic style so points remain visible even if
                // geometryDimension semantics vary across vector tile sources.
                return {
                    radius: getPointRadius(z),
                    fill: true,
                    fillColor: fillColor,
                    fillOpacity: 0.75,
                    color: '#0f172a',
                    weight: 0.8,
                    opacity: 0.95,
                };
            };

            try {
                scenarioDatasetKeys.forEach((scenarioKey) => {
                    const entry = vectorTileIndex[scenarioKey];
                    if (!entry?.url_template) return;

                    const layerName = entry.layer || 'sites';
                    const tileUrl = DATA_PATH + entry.url_template;
                    const vectorLayer = L.vectorGrid.protobuf(tileUrl, {
                        interactive: true,
                        pane: 'sitePointPane',
                        maxNativeZoom: Number(entry.max_zoom || 14),
                        vectorTileLayerStyles: {
                            [layerName]: styleByGeometry
                        }
                    });

                    vectorLayer.on('click', (e) => {
                        const props = e.layer?.properties || {};
                        const lossFraction = Number(props.loss_fraction || 0);
                        const lossValue = isCumulative
                            ? Number(props.cumulative_loss || 0)
                            : Number(props.value_loss || props.annual_loss || 0);
                        const popupContent = `
                            <div class="popup-content">
                                <strong>${props.country || 'Unknown'}</strong><br>
                                Dataset: ${formatValueType(props.value_type || 'unknown')}<br>
                                <span style="color: #94a3b8;">${describeValueType(props.value_type)}</span><br>
                                Original Value (annual): ${formatMoney(Number(props.original_value || 0))}<br>
                                ${lossType} Loss: ${formatMoney(lossValue)} (${(lossFraction * 100).toFixed(1)}%)<br>
                                Coral change: ${(Number(props.coral_change || 0) * 100).toFixed(1)}pp
                            </div>
                        `;
                        L.popup().setLatLng(e.latlng).setContent(popupContent).openOn(map);
                    });

                    siteLayer.addLayer(vectorLayer);
                });

                updateMapLegend(
                    scale,
                    showChoropleth,
                    null,
                    'Vector tiles mode (zoom/pan loaded natively)'
                );

                if (!preserveView) {
                    map.setView(map.getCenter(), zoom);
                }
                if (showChoropleth) {
                    renderChoropleth();
                }
                bringTourismLayersToFront();
            } catch (error) {
                console.error('Error rendering vector tiles:', error);
            } finally {
                isRendering = false;
            }
        }
        
        function renderSites(geojson, isCumulative = false, preserveView = null) {
            if (!geojson || !geojson.features || geojson.features.length === 0) {
                console.warn('No features to render');
                clearSiteDataLayers();
                isRendering = false;
                return;
            }
            
            if (!map || !siteLayer) {
                console.error('Map or siteLayer not initialized');
                isRendering = false;
                return;
            }
            
            // Prevent concurrent renders
            if (isRendering) {
                return;
            }
            isRendering = true;
            
            clearSiteDataLayers();
            currentSiteGeojson = geojson;
            
            const metric = document.getElementById('map-metric').value;
            const scale = getColorScale(isCumulative, metric);
            const showChoropleth = document.getElementById('map-choropleth-toggle').checked;
            
            // Get current zoom level
            const zoom = map.getZoom();
            
            // Filter features by viewport (only render visible features)
            const mapBounds = map.getBounds();
            const filteredGeoJSON = filterFeaturesByViewport(geojson, mapBounds);
            
            console.log(`Filtered ${geojson.features.length} features to ${filteredGeoJSON.features.length} visible features`);

            const isPointGeometry = (geometry) => {
                const type = geometry?.type;
                return type === 'Point' || type === 'MultiPoint';
            };

            const getFeatureColorValue = (props) => {
                const lossFraction = props.loss_fraction || 0;
                const lossValue = isCumulative
                    ? (props.cumulative_loss || 0)
                    : (props.value_loss || props.annual_loss || 0);
                return metric === 'loss_percent' ? lossFraction : lossValue;
            };

            const pointValues = filteredGeoJSON.features
                .filter((feature) => isPointGeometry(feature.geometry))
                .map((feature) => getFeatureColorValue(feature.properties || {}))
                .filter((value) => Number.isFinite(value));
            const pointValueTypes = [
                ...new Set(
                    filteredGeoJSON.features
                        .filter((feature) => isPointGeometry(feature.geometry))
                        .map((feature) => feature?.properties?.value_type)
                        .filter(Boolean)
                ),
            ];
            const pointRange = pointValues.length
                ? { min: Math.min(...pointValues), max: Math.max(...pointValues) }
                : null;
            const pointDatasetMessage = pointValueTypes.length
                ? `Points represent: ${pointValueTypes.map((vt) => formatValueType(vt)).join(', ')}`
                : '';
            
            // Update legend
            updateMapLegend(
                scale,
                showChoropleth,
                pointRange,
                pointValues.length
                    ? pointDatasetMessage
                    : (pointDatasetMessage || 'No point features in current viewport')
            );
            
            // Format values for popup
            const formatMoney = (val) => {
                if (val >= 1e6) return `$${(val / 1e6).toFixed(2)}M`;
                if (val >= 1e3) return `$${(val / 1e3).toFixed(1)}k`;
                return `$${val.toFixed(0)}`;
            };
            
            const lossType = isCumulative ? 'Cumulative' : 'Annual';
            
            // Create style function that will be applied to each feature
            const styleFunction = (feature) => {
                const props = feature.properties;
                
                // Get loss data based on cumulative vs annual
                const lossFraction = props.loss_fraction || 0;
                const lossValue = isCumulative ?
                    (props.cumulative_loss || 0) :
                    (props.value_loss || props.annual_loss || 0);
                
                // Determine color value based on metric
                let colorValue;
                if (metric === 'loss_percent') {
                    colorValue = lossFraction;
                } else {
                    colorValue = lossValue;
                }
                
                const fillColor = getColorFromScale(colorValue, scale);
                
                return {
                    pane: 'sitePolygonPane',
                    renderer: siteTourismRenderer,
                    fillColor: fillColor,
                    color: '#1a2332',
                    weight: zoom < 5 ? 0.5 : 1,  // Thinner lines at low zoom
                    opacity: 0.9,
                    fillOpacity: 0.7
                };
            };

            const DATASET_FILL_OPACITY = { coastal_protection: 0.65, fisheries: 0.75, tourism: 0.88 };

            const pointToLayer = (feature, latlng) => {
                const props = feature.properties || {};
                const colorValue = getFeatureColorValue(props);
                const fillColor = getColorFromScale(colorValue, scale);
                const gridRes = Number(props.grid_resolution_deg);
                if (Number.isFinite(gridRes) && gridRes > 0) {
                    // Fisheries/coastal-protection only: exact-fit rectangles in siteBackPane.
                    const half = gridRes / 2;
                    const bounds = [
                        [latlng.lat - half, latlng.lng - half],
                        [latlng.lat + half, latlng.lng + half],
                    ];
                    return L.rectangle(bounds, {
                        pane: 'siteBackPane',
                        renderer: siteGridRenderer,
                        fillColor,
                        color: 'transparent',
                        weight: 0,
                        fillOpacity: DATASET_FILL_OPACITY[props.value_type] ?? 0.72,
                    });
                }
                const aligned = applyPointAlignmentOffset(latlng);
                return L.circleMarker(aligned, {
                    pane: 'sitePointPane',
                    radius: getPointRadius(zoom),
                    fillColor,
                    color: '#0f172a',
                    weight: 0.8,
                    opacity: 0.95,
                    fillOpacity: 0.8,
                    stroke: false
                });
            };
            
            // Create popup function
            const onEachFeature = (feature, layer) => {
                const props = feature.properties;
                const lossFraction = props.loss_fraction || 0;
                const lossValue = isCumulative ?
                    (props.cumulative_loss || 0) :
                    (props.value_loss || props.annual_loss || 0);
                
                // const layerLabel = props.n_sites > 1
                //     ? `Grid cell (${props.n_sites} sites)`
                //     : (isPointGeometry(feature.geometry) ? 'Point layer' : 'Polygon layer');
                const popupContent = `
                    <div class="popup-content">
                        <strong>${props.country || 'Unknown'}</strong><br>
                        Dataset: ${formatValueType(props.value_type || 'unknown')}<br>
                        <span style="color: #94a3b8;">${describeValueType(props.value_type)}</span><br>
                        Original Value (annual): ${formatMoney(props.original_value || 0)}<br>
                        ${lossType} Loss: ${formatMoney(lossValue)} (${(lossFraction * 100).toFixed(1)}%)<br>
                        Coral change: ${((props.coral_change || 0) * 100).toFixed(1)}pp
                    </div>
                `;
                layer.bindPopup(popupContent);
            };
            
            // Only simplify at very low zoom levels (zoom < 3) to preserve shape quality
            // Use filtered GeoJSON directly without simplification for normal zoom levels
            const simplifiedGeoJSON = zoom < 3 ? {
                type: 'FeatureCollection',
                features: filteredGeoJSON.features.map(feature => ({
                    ...feature,
                    geometry: simplifyGeometry(feature.geometry, 0.001)  // Only at very low zoom
                }))
            } : filteredGeoJSON;
            
            // Split grid points and tourism polygons into separate layer groups so
            // pane/renderer z-order is respected (canvas batching otherwise stacks
            // grid cells above reef polygons).
            const gridFeatures = simplifiedGeoJSON.features.filter((f) =>
                isPointGeometry(f.geometry)
            );
            const tourismFeatures = simplifiedGeoJSON.features.filter((f) =>
                !isPointGeometry(f.geometry)
            );

            // Render features using requestAnimationFrame for smooth rendering
            requestAnimationFrame(() => {
                try {
                    let layerCount = 0;

                    if (gridFeatures.length > 0) {
                        const gridGeoJson = L.geoJSON(
                            { type: 'FeatureCollection', features: gridFeatures },
                            { pointToLayer, onEachFeature }
                        );
                        gridGeoJson.eachLayer((layer) => {
                            siteGridLayer.addLayer(layer);
                            layerCount++;
                        });
                    }

                    if (tourismFeatures.length > 0) {
                        const tourismGeoJson = L.geoJSON(
                            { type: 'FeatureCollection', features: tourismFeatures },
                            { style: styleFunction, onEachFeature }
                        );
                        tourismGeoJson.eachLayer((layer) => {
                            siteTourismLayer.addLayer(layer);
                            layerCount++;
                        });
                    }

                    bringTourismLayersToFront();
                    
                    console.log(`✓ Added ${layerCount} layers to map (simplified, zoom: ${zoom})`);
                    
                    // Only fit bounds on initial load, otherwise preserve current view
                    if (layerCount > 0 && !preserveView) {
                        // Initial load: fit bounds to show all polygons
                        try {
                            const bounds = L.geoJSON(geojson).getBounds();
                            if (bounds && bounds.isValid()) {
                                map.fitBounds(bounds, { 
                                    padding: [50, 50],
                                    maxZoom: 10
                                });
                                console.log(`✓ Map bounds set to show all polygons`);
                            } else {
                                console.warn('Invalid bounds, using default view');
                                map.setView([0, 0], 2);
                            }
                        } catch (boundsError) {
                            console.warn('Error setting map bounds:', boundsError);
                            map.setView([0, 0], 2);
                        }
                    } else if (preserveView) {
                        const currentCenter = map.getCenter();
                        const centerChanged =
                            Math.abs(currentCenter.lat - preserveView.center.lat) > 1e-9 ||
                            Math.abs(currentCenter.lng - preserveView.center.lng) > 1e-9;
                        const zoomChanged = map.getZoom() !== preserveView.zoom;
                        if (centerChanged || zoomChanged) {
                            map.setView(preserveView.center, preserveView.zoom);
                        }
                    }
                    
                    // Update choropleth if it's enabled
                    if (showChoropleth) {
                        renderChoropleth();
                    }
                    // Ensure points stay on top after any redraw.
                    bringTourismLayersToFront();
                } catch (error) {
                    console.error('Error rendering sites:', error);
                } finally {
                    isRendering = false;
                }
            });
        }

        function toggleChoropleth() {
            const enabled = document.getElementById('map-choropleth-toggle').checked;
            const scenario = document.getElementById('map-scenario').value;
            const isCumulative = scenario.startsWith('cumulative_');
            const metric = document.getElementById('map-metric').value;
            const scale = getColorScale(isCumulative, metric);
            
            if (enabled) {
                renderChoropleth();
            } else {
                if (choroplethLayer) {
                    map.removeLayer(choroplethLayer);
                    choroplethLayer = L.layerGroup();
                }
                // Update legend without choropleth indicator
                updateMapLegend(scale, false, null, '');
                removeChoroplethLegend();
            }
        }
        
        function renderChoropleth() {
            const scenario = document.getElementById('map-scenario').value;
            const isCumulative = scenario.startsWith('cumulative_');
            const metric = document.getElementById('map-metric').value;
            const selectedValueTypes = getMapSelectedValueTypes();
            const dataSource = isCumulative ? cumulativeCountryData : countryData;
            if (selectedValueTypes.length === 0) {
                if (choroplethLayer) {
                    map.removeLayer(choroplethLayer);
                    choroplethLayer = L.layerGroup();
                }
                return;
            }
            
            if (!dataSource || !countryBoundaries) {
                console.warn('Country data or boundaries not loaded. dataSource:', !!dataSource, 'countryBoundaries:', !!countryBoundaries, 'isCumulative:', isCumulative);
                if (isCumulative) {
                    console.log('cumulativeCountryData:', cumulativeCountryData ? `loaded (${cumulativeCountryData.length} records)` : 'not loaded');
                } else {
                    console.log('countryData:', countryData ? `loaded (${countryData.length} records)` : 'not loaded');
                }
                return;
            }
            
            const model = document.getElementById('map-model').value;
            const scale = getColorScale(isCumulative, metric);
            
            console.log('Rendering choropleth for scenario:', scenario, 'model:', model, 'isCumulative:', isCumulative, 'metric:', metric);
            
            // Update legend with choropleth indicator
            updateMapLegend(scale, true);
            
            // Normalize model names for matching
            const normalizeModelName = (name) => {
                return name.toLowerCase()
                    .replace(/\s+/g, '_')
                    .replace(/\//g, '_')
                    .replace(/%/g, 'pct')
                    .replace(/[()]/g, '')
                    .replace(/=/g, '');
            };
            
            // Filter country data for current scenario and model
            const filtered = dataSource.filter(c => {
                const cScenario = c.scenario.toLowerCase();
                const targetScenario = scenario.toLowerCase();
                const cModel = normalizeModelName(c.model);
                const targetModel = normalizeModelName(model);
                const scenarioMatch = cScenario === targetScenario;
                const modelMatch = cModel === targetModel;
                const valueTypeMatch = selectedValueTypes.includes(c.value_type);
                return scenarioMatch && modelMatch && valueTypeMatch;
            });
            const filteredData = selectedValueTypes.length > 1
                ? aggregateRowsByCountry(filtered.map(r => ({ ...r, scenario, model })), isCumulative, false)
                : filtered;
            
            console.log('Filtered countries:', filteredData.length);
            
            // Create lookup map by country name
            const countryMetrics = {};
            filteredData.forEach(c => {
                const countryName = c.country;
                // Store by multiple possible keys for flexible matching
                countryMetrics[countryName] = {
                    value_loss: isCumulative ? (c.cumulative_loss || 0) : (c.value_loss || 0),
                    loss_fraction: c.loss_fraction || 0, // Always use annual loss fraction
                    iso_a3: c.iso_a3 || ''
                };
                // Also try common name variations
                if (countryName.includes(',')) {
                    countryMetrics[countryName.split(',')[0].trim()] = countryMetrics[countryName];
                }
            });
            
            // Remove existing choropleth
            if (choroplethLayer) {
                map.removeLayer(choroplethLayer);
            }
            choroplethLayer = L.layerGroup();
            
            // Create styled GeoJSON layer
            const styledGeoJson = L.geoJSON(countryBoundaries, {
                pane: 'choroplethPane',
                style: (feature) => {
                    const countryName = feature.properties.name;
                    // Try exact match first, then try partial matches
                    let countryMetric = countryMetrics[countryName];
                    if (!countryMetric) {
                        // Try matching by partial name
                        for (const [key, value] of Object.entries(countryMetrics)) {
                            if (countryName.includes(key) || key.includes(countryName)) {
                                countryMetric = value;
                                break;
                            }
                        }
                    }
                    
                    // Determine color value based on metric
                    let colorValue = null;
                    if (countryMetric) {
                        if (metric === 'loss_percent') {
                            colorValue = countryMetric.loss_fraction;
                        } else {
                            colorValue = countryMetric.value_loss;
                        }
                    }
                    
                    const fillColor = colorValue !== null ? getColorFromScale(colorValue, scale) : '#64748b';
                    
                    return {
                        fillColor: fillColor,
                        weight: 1,
                        opacity: 0.8,
                        color: '#1a2332',
                        fillOpacity: countryMetric ? 0.6 : 0,
                        dashArray: countryMetric ? null : '5, 5'
                    };
                },
                onEachFeature: (feature, layer) => {
                    const countryName = feature.properties.name;
                    // Try exact match first, then try partial matches
                    let countryMetric = countryMetrics[countryName];
                    if (!countryMetric) {
                        for (const [key, value] of Object.entries(countryMetrics)) {
                            if (countryName.includes(key) || key.includes(countryName)) {
                                countryMetric = value;
                                break;
                            }
                        }
                    }
                    
                    if (countryMetric) {
                        const lossType = isCumulative ? 'Cumulative' : 'Annual';
                        const formatMoney = (val) => {
                            if (val >= 1e9) return `$${(val / 1e9).toFixed(2)}B`;
                            if (val >= 1e6) return `$${(val / 1e6).toFixed(1)}M`;
                            if (val >= 1e3) return `$${(val / 1e3).toFixed(1)}k`;
                            return `$${val.toFixed(0)}`;
                        };
                        
                        const popupContent = `
                            <div class="popup-content">
                                <strong>${countryName}</strong><br>
                                ${lossType} Loss: ${formatMoney(countryMetric.value_loss)}<br>
                                Loss: ${(countryMetric.loss_fraction * 100).toFixed(1)}%
                            </div>
                        `;
                        layer.bindPopup(popupContent);
                    }
                }
            });
            
            choroplethLayer.addLayer(styledGeoJson);
            choroplethLayer.addTo(map);
            // Reassert point precedence after choropleth draw.
            bringTourismLayersToFront();
            
            // Add or update legend control
            addChoroplethLegend();
        }
        
        let choroplethLegendControl = null;
        
        function addChoroplethLegend() {
            // Remove existing legend if any
            if (choroplethLegendControl) {
                map.removeControl(choroplethLegendControl);
            }
            
            choroplethLegendControl = L.control({position: 'bottomright'});
            
            choroplethLegendControl.onAdd = function(map) {
                const div = L.DomUtil.create('div', 'choropleth-legend');
                div.style.cssText = 'background: none; padding: 0; border: none; font-family: Instrument Sans, sans-serif; font-size: 12px; color: #64748b;';
                
                // Only the text at the bottom (no colored boxes or label)
                div.innerHTML = '<div style="color: #64748b; font-size: 11px; padding: 0.25em 0.5em; background: none;">Dashed = No data</div>';
                
                return div;
            };
            
            choroplethLegendControl.addTo(map);
        }
        
        function removeChoroplethLegend() {
            if (choroplethLegendControl) {
                map.removeControl(choroplethLegendControl);
                choroplethLegendControl = null;
            }
        }

        // ============================================================
        // INITIALIZE
        // ============================================================
        
        document.addEventListener('DOMContentLoaded', loadData);

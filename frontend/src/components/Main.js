import React, { useEffect, useState } from 'react';
import Axios from 'axios';
import Button from '@material-ui/core/Button';
import { CircularProgress, Typography } from '@material-ui/core';
import { withStyles } from '@material-ui/core/styles';
import MaterialTable from 'material-table';
import SearchIcon from '@material-ui/icons/Search';

const styles = {
	predictButton: {
		width: '12rem',
	},
};

const getColumns = keys =>
	keys
		.map(key => (key !== 'tableData' ? { title: key, field: key } : undefined))
		.filter(val => val !== undefined);

const Main = ({ classes, datasetId }) => {
	const [dataset, setDataset] = useState(undefined);

	const [predicting, setPredicting] = useState(false);
	const [prediction, setPrediction] = useState('');

	const [retrieving, setRetrieving] = useState(false);
	const [allSamples, setSamples] = useState([]);

	useEffect(() => {
		if (datasetId != null) getDataset(datasetId);
		setPrediction('');
		setSamples([]);
	}, [datasetId]);

	const getDataset = id => {
		Axios.get(`/api/datasets/${id}`)
			.then(res => setDataset(res.data))
			.catch(err => console.log(err));
	};

	const retrieveSamples = () => {
		if (retrieving) return;
		setRetrieving(true);
		Axios.get(`/api/datasets/${datasetId}/retrieve_samples/`)
			.then(res => {
				setSamples(res.data);
				setRetrieving(false);
			})
			.catch(err => {
				console.log(err);
				setRetrieving(false);
			});
	};

	const predictSample = (sampleId = -1) => {
		if (predicting) return;
		setPredicting(true);
		Axios.get(`/api/datasets/${datasetId}/predict_sample/`, { params: { sampleId } })
			.then(res => {
				setPrediction(res.data);
				setPredicting(false);
			})
			.catch(err => {
				console.log(err);
				setPredicting(false);
			});
	};

	return dataset ? (
		<React.Fragment>
			<Typography variant="h1">{dataset.title}</Typography>
			<Button
				variant="contained"
				color="primary"
				onClick={() => retrieveSamples()}
				className={classes.retrieveButton}
			>
				{retrieving ? (
					<CircularProgress size={26} color="white" />
				) : (
					<Typography>Retrieve samples</Typography>
				)}
			</Button>
			{allSamples.length > 0 && (
				<MaterialTable
					title={dataset.title}
					data={allSamples}
					columns={getColumns(Object.keys(allSamples[0]))}
					actions={[
						{
							icon: () => <SearchIcon />,
							tooltip: 'Predict sample',
							onClick: (event, rowData) => predictSample(rowData.tableData.id),
						},
					]}
				/>
			)}
			<Button
				variant="contained"
				color="primary"
				onClick={() => predictSample()}
				className={classes.predictButton}
			>
				{predicting ? (
					<CircularProgress size={26} color="white" />
				) : (
					<Typography>Predict a sample</Typography>
				)}
			</Button>
			{prediction !== '' && <h2>{prediction}</h2>}
		</React.Fragment>
	) : (
		<CircularProgress />
	);
};

export default withStyles(styles)(Main);

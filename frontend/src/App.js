import React, { useState } from 'react';

import Main from './components/Main';

import { makeStyles } from '@material-ui/core/styles';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';

const useStyles = makeStyles(theme => ({
	root: {
		flexGrow: 1,
	},
	menuButton: {
		marginRight: theme.spacing(2),
	},
	title: {
		flexGrow: 1,
	},
}));

const App = () => {
	const classes = useStyles();
	const [datasetId, setDatasetId] = useState(1590);

	return (
		<div className={classes.root}>
			<AppBar position="static">
				<Toolbar>
					{/* <IconButton edge="start" className={classes.menuButton} color="inherit" aria-label="menu">
						<MenuIcon />
					</IconButton> */}
					<Typography variant="h6" className={classes.title}>
						Constrastive explanations
					</Typography>
					<Button onClick={() => setDatasetId(1590)} color="inherit">
						Adult
					</Button>
					<Button onClick={() => setDatasetId(31)} color="inherit">
						Credit-g
					</Button>
				</Toolbar>
			</AppBar>
			<Main datasetId={datasetId} />
		</div>
	);
};

export default App;

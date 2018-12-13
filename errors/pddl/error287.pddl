(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b0)
		(clear b0)
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(on-table b3)
		(clear b3)
		(on-table b4)
		(clear b4)
		(on-table b5)
		(clear b5)
		(on-table b8)
		(clear b8)
		(clear b9)
		(in-tower t0)
		(on b7 t0)
		(in-tower b7)
		(on b6 b7)
		(in-tower b6)
		(on b9 b6)
		(in-tower b9)
		(green b0)
		(blue b0)
		(blue b1)
		(yellow b4)
		(blue b5)
		(green b6)
		(red b6)
		(blue b6)
		(green b7)
		(red b9)
		(yellow b3)
		(yellow b2)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (green ?x)) (exists (?y) (and (yellow ?y) (on ?x ?y))))) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b9 t0)) (not (on b8 t0)) (not (on b9 b7)) (not (on b8 b7)) (not (on b8 b9)) (not (on b5 b9)))))
)